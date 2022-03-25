# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for tfx.orchestration.experimental.core.pipeline_ops."""

import copy
import os
import threading
import time

from absl.testing import parameterized
from absl.testing.absltest import mock
import tensorflow as tf
from tfx import types
from tfx.dsl.compiler import constants
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import async_pipeline_task_gen
from tfx.orchestration.experimental.core import env
from tfx.orchestration.experimental.core import mlmd_state
from tfx.orchestration.experimental.core import orchestration_options
from tfx.orchestration.experimental.core import pipeline_ops
from tfx.orchestration.experimental.core import pipeline_state as pstate
from tfx.orchestration.experimental.core import service_jobs
from tfx.orchestration.experimental.core import sync_pipeline_task_gen
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_gen_utils
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.experimental.core import test_utils
from tfx.orchestration.experimental.core.task_schedulers import manual_task_scheduler
from tfx.orchestration.experimental.core.testing import test_async_pipeline
from tfx.orchestration.experimental.core.testing import test_manual_node
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable import outputs_utils
from tfx.orchestration.portable import partial_run_utils
from tfx.orchestration.portable import runtime_parameter_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import status as status_lib

from ml_metadata.proto import metadata_store_pb2
from ml_metadata.proto import metadata_store_service_pb2


def _test_pipeline(pipeline_id: str,
                   execution_mode: pipeline_pb2.Pipeline.ExecutionMode = (
                       pipeline_pb2.Pipeline.ASYNC)):
  pipeline = pipeline_pb2.Pipeline()
  pipeline.pipeline_info.id = pipeline_id
  pipeline.execution_mode = execution_mode
  if execution_mode == pipeline_pb2.Pipeline.SYNC:
    pipeline.runtime_spec.pipeline_run_id.field_value.string_value = 'run0'
  return pipeline


def _test_pipeline_with_example_gen(pipeline_id: str):
  pipeline = _test_pipeline(pipeline_id, pipeline_pb2.Pipeline.SYNC)
  node_example_gen = pipeline.nodes.add().pipeline_node
  node_example_gen.node_info.id = 'example-gen-0'
  node_example_gen.node_info.type.name = 'ExampleGen'
  context_spec = node_example_gen.contexts.contexts.add()
  context_spec.name.field_value.string_value = 'example-gen-0'
  context_spec.type.name = 'node'
  return pipeline


class PipelineOpsTest(test_utils.TfxTest, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    pipeline_root = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self.id())

    # Makes sure multiple connections within a test always connect to the same
    # MLMD instance.
    metadata_path = os.path.join(pipeline_root, 'metadata', 'metadata.db')
    self._metadata_path = metadata_path
    connection_config = metadata.sqlite_metadata_connection_config(
        metadata_path)
    connection_config.sqlite.SetInParent()
    self._mlmd_connection = metadata.Metadata(
        connection_config=connection_config)

    mock_service_job_manager = mock.create_autospec(
        service_jobs.ServiceJobManager, instance=True)
    mock_service_job_manager.is_pure_service_node.side_effect = (
        lambda _, node_id: node_id == 'ExampleGen')
    mock_service_job_manager.is_mixed_service_node.side_effect = (
        lambda _, node_id: node_id == 'Transform')
    mock_service_job_manager.stop_node_services.return_value = True
    self._mock_service_job_manager = mock_service_job_manager

  @parameterized.named_parameters(
      dict(testcase_name='async', pipeline=_test_pipeline('pipeline1')),
      dict(
          testcase_name='sync',
          pipeline=_test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC)))
  def test_initiate_pipeline_start(self, pipeline):
    with self._mlmd_connection as m:
      # Initiate a pipeline start.
      with pipeline_ops.initiate_pipeline_start(m, pipeline) as pipeline_state1:
        self.assertProtoPartiallyEquals(
            pipeline, pipeline_state1.pipeline, ignored_fields=['runtime_spec'])
        self.assertEqual(metadata_store_pb2.Execution.NEW,
                         pipeline_state1.get_pipeline_execution_state())

      # Initiate another pipeline start.
      pipeline2 = _test_pipeline('pipeline2')
      with pipeline_ops.initiate_pipeline_start(m,
                                                pipeline2) as pipeline_state2:
        self.assertEqual(pipeline2, pipeline_state2.pipeline)
        self.assertEqual(metadata_store_pb2.Execution.NEW,
                         pipeline_state2.get_pipeline_execution_state())

      # Error if attempted to initiate when old one is active.
      with self.assertRaises(status_lib.StatusNotOkError) as exception_context:
        pipeline_ops.initiate_pipeline_start(m, pipeline)
      self.assertEqual(status_lib.Code.ALREADY_EXISTS,
                       exception_context.exception.code)

      # Fine to initiate after the previous one is inactive.
      with pipeline_state1:
        pipeline_state1.set_pipeline_execution_state(
            metadata_store_pb2.Execution.COMPLETE)
      with pipeline_ops.initiate_pipeline_start(m, pipeline) as pipeline_state3:
        self.assertEqual(metadata_store_pb2.Execution.NEW,
                         pipeline_state3.get_pipeline_execution_state())

  @mock.patch.object(partial_run_utils, 'snapshot')
  def test_resume_pipeline(self, mock_snapshot):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('test_pipeline', pipeline_pb2.Pipeline.SYNC)
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      node_example_gen = pipeline.nodes.add().pipeline_node
      node_example_gen.node_info.id = 'ExampleGen'
      node_example_gen.downstream_nodes.extend(['Trainer'])
      node_trainer = pipeline.nodes.add().pipeline_node
      node_trainer.node_info.id = 'Trainer'
      node_trainer.upstream_nodes.extend(['ExampleGen'])

      # Error if attempt to resume the pipeline when there is no previous run.
      with self.assertRaises(status_lib.StatusNotOkError) as exception_context:
        pipeline_ops.resume_pipeline(m, pipeline)
      self.assertEqual(status_lib.Code.NOT_FOUND,
                       exception_context.exception.code)

      # Initiate a pipeline start.
      pipeline_state_run1 = pipeline_ops.initiate_pipeline_start(m, pipeline)

      # Error if attempt to resume the pipeline when the previous one is active.
      pipeline.runtime_spec.pipeline_run_id.field_value.string_value = 'run1'
      with self.assertRaises(status_lib.StatusNotOkError) as exception_context:
        pipeline_ops.resume_pipeline(m, pipeline)
      self.assertEqual(status_lib.Code.ALREADY_EXISTS,
                       exception_context.exception.code)

      with pipeline_state_run1:
        example_gen_node_uid = task_lib.NodeUid(pipeline_uid, 'ExampleGen')
        trainer_node_uid = task_lib.NodeUid(pipeline_uid, 'Trainer')
        with pipeline_state_run1.node_state_update_context(
            example_gen_node_uid) as node_state:
          node_state.update(pstate.NodeState.COMPLETE)
        with pipeline_state_run1.node_state_update_context(
            trainer_node_uid) as node_state:
          node_state.update(pstate.NodeState.FAILED)
        pipeline_state_run1.set_pipeline_execution_state(
            metadata_store_pb2.Execution.COMPLETE)
        pipeline_state_run1.initiate_stop(
            status_lib.Status(code=status_lib.Code.ABORTED))

      # Only Trainer is marked to run since ExampleGen succeeded in previous
      # run.
      expected_pipeline = copy.deepcopy(pipeline)
      partial_run_utils.set_latest_pipeline_run_strategy(
          expected_pipeline.runtime_spec.snapshot_settings)
      expected_pipeline.nodes[
          0].pipeline_node.execution_options.skip.reuse_artifacts_mode = pipeline_pb2.NodeExecutionOptions.Skip.REQUIRED
      expected_pipeline.nodes[
          1].pipeline_node.execution_options.run.perform_snapshot = True
      expected_pipeline.nodes[
          1].pipeline_node.execution_options.run.depends_on_snapshot = True
      with pipeline_ops.resume_pipeline(m, pipeline) as pipeline_state_run2:
        self.assertEqual(expected_pipeline, pipeline_state_run2.pipeline)
        pipeline_state_run2.is_active()
        mock_snapshot.assert_called_once()

  @mock.patch.object(partial_run_utils, 'snapshot')
  def test_initiate_pipeline_start_with_invalid_partial_run(
      self, mock_snapshot):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('test_pipeline', pipeline_pb2.Pipeline.SYNC)
      node_example_gen = pipeline.nodes.add().pipeline_node
      node_example_gen.node_info.id = 'ExampleGen'
      node_example_gen.downstream_nodes.extend(['Transform'])
      node_transform = pipeline.nodes.add().pipeline_node
      node_transform.node_info.id = 'Transform'
      node_transform.upstream_nodes.extend(['ExampleGen'])
      node_transform.downstream_nodes.extend(['Trainer'])
      node_trainer = pipeline.nodes.add().pipeline_node
      node_trainer.node_info.id = 'Trainer'
      node_trainer.upstream_nodes.extend(['Transform'])

      incorrect_partial_run_option = pipeline_pb2.PartialRun(
          from_nodes=['InvalidaNode'],
          to_nodes=['Trainer'],
          snapshot_settings=partial_run_utils.latest_pipeline_snapshot_settings(
          ))
      with self.assertRaisesRegex(
          status_lib.StatusNotOkError,
          'specified in from_nodes/to_nodes are not present in the pipeline.'):
        pipeline_ops.initiate_pipeline_start(
            m, pipeline, partial_run_option=incorrect_partial_run_option)

  @mock.patch.object(partial_run_utils, 'snapshot')
  def test_initiate_pipeline_start_with_partial_run(self, mock_snapshot):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('test_pipeline', pipeline_pb2.Pipeline.SYNC)
      node_example_gen = pipeline.nodes.add().pipeline_node
      node_example_gen.node_info.id = 'ExampleGen'
      node_example_gen.downstream_nodes.extend(['Transform'])
      node_transform = pipeline.nodes.add().pipeline_node
      node_transform.node_info.id = 'Transform'
      node_transform.upstream_nodes.extend(['ExampleGen'])
      node_transform.downstream_nodes.extend(['Trainer'])
      node_trainer = pipeline.nodes.add().pipeline_node
      node_trainer.node_info.id = 'Trainer'
      node_trainer.upstream_nodes.extend(['Transform'])

      expected_pipeline = copy.deepcopy(pipeline)
      partial_run_utils.set_latest_pipeline_run_strategy(
          expected_pipeline.runtime_spec.snapshot_settings)
      expected_pipeline.nodes[
          0].pipeline_node.execution_options.skip.reuse_artifacts_mode = pipeline_pb2.NodeExecutionOptions.Skip.REQUIRED
      expected_pipeline.nodes[
          1].pipeline_node.execution_options.run.perform_snapshot = True
      expected_pipeline.nodes[
          1].pipeline_node.execution_options.run.depends_on_snapshot = True
      expected_pipeline.nodes[
          2].pipeline_node.execution_options.run.SetInParent()

      partial_run_option = pipeline_pb2.PartialRun(
          from_nodes=['Transform'],
          to_nodes=['Trainer'],
          snapshot_settings=partial_run_utils.latest_pipeline_snapshot_settings(
          ))
      with pipeline_ops.initiate_pipeline_start(
          m, pipeline, partial_run_option=partial_run_option) as pipeline_state:
        mock_snapshot.assert_called_once()
        self.assertEqual(expected_pipeline, pipeline_state.pipeline)

  @mock.patch.object(partial_run_utils, 'snapshot')
  def test_partial_run_with_previously_failed_nodes(self, mock_snapshot):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('test_pipeline', pipeline_pb2.Pipeline.SYNC)
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      node_example_gen = pipeline.nodes.add().pipeline_node
      node_example_gen.node_info.id = 'ExampleGen'
      node_example_gen.downstream_nodes.extend(['Transform', 'Trainer'])
      node_transform = pipeline.nodes.add().pipeline_node
      node_transform.node_info.id = 'Transform'
      node_transform.upstream_nodes.extend(['ExampleGen'])
      node_trainer = pipeline.nodes.add().pipeline_node
      node_trainer.node_info.id = 'Trainer'
      node_trainer.upstream_nodes.extend(['ExampleGen'])

      example_gen_node_uid = task_lib.NodeUid(pipeline_uid, 'ExampleGen')
      trainer_node_uid = task_lib.NodeUid(pipeline_uid, 'Trainer')
      transform_node_uid = task_lib.NodeUid(pipeline_uid, 'Transform')

      def _stop_pipeline(pipeline_state):
        pipeline_state.set_pipeline_execution_state(
            metadata_store_pb2.Execution.COMPLETE)
        pipeline_state.initiate_stop(
            status_lib.Status(code=status_lib.Code.ABORTED))

      # In run0, trainer and transform failed.
      with pipeline_ops.initiate_pipeline_start(
          m, pipeline) as pipeline_state_run0:
        with pipeline_state_run0.node_state_update_context(
            example_gen_node_uid) as node_state:
          node_state.update(pstate.NodeState.COMPLETE)
        with pipeline_state_run0.node_state_update_context(
            trainer_node_uid) as node_state:
          node_state.update(pstate.NodeState.FAILED)
        with pipeline_state_run0.node_state_update_context(
            transform_node_uid) as node_state:
          node_state.update(pstate.NodeState.FAILED)
        _stop_pipeline(pipeline_state_run0)

      # Partial run based on run0, trainer is skipped and state indicates that
      # it failed previously. Only transform runs and it fails again.
      partial_run_option = pipeline_pb2.PartialRun(
          from_nodes=['Transform'], to_nodes=['Transform'])
      pipeline.runtime_spec.pipeline_run_id.field_value.string_value = 'run1'
      with pipeline_ops.initiate_pipeline_start(
          m, pipeline,
          partial_run_option=partial_run_option) as pipeline_state_run1:
        self.assertEqual(
            pipeline_state_run1.get_node_state(trainer_node_uid).state,
            pstate.NodeState.SKIPPED_PARTIAL_RUN)
        self.assertEqual(
            pipeline_state_run1.get_node_state(
                trainer_node_uid, pstate._PREVIOUS_NODE_STATES).state,
            pstate.NodeState.FAILED)
        self.assertEqual(
            pipeline_state_run1.get_node_state(example_gen_node_uid).state,
            pstate.NodeState.SKIPPED_PARTIAL_RUN)
        self.assertEqual(
            pipeline_state_run1.get_node_state(
                example_gen_node_uid, pstate._PREVIOUS_NODE_STATES).state,
            pstate.NodeState.COMPLETE)
        self.assertEqual(
            pipeline_state_run1.get_node_state(transform_node_uid).state,
            pstate.NodeState.STARTED)

        with pipeline_state_run1.node_state_update_context(
            transform_node_uid) as node_state:
          node_state.update(pstate.NodeState.FAILED)
        _stop_pipeline(pipeline_state_run1)

      # Partial run based on run1, trainer and transform are skipped and
      # correctly indicate they've failed previously.
      partial_run_option = pipeline_pb2.PartialRun(
          from_nodes=['ExampleGen'], to_nodes=['ExampleGen'])
      pipeline.runtime_spec.pipeline_run_id.field_value.string_value = 'run2'
      with pipeline_ops.initiate_pipeline_start(
          m, pipeline,
          partial_run_option=partial_run_option) as pipeline_state_run2:
        self.assertEqual(
            pipeline_state_run2.get_node_state(trainer_node_uid).state,
            pstate.NodeState.SKIPPED_PARTIAL_RUN)
        self.assertEqual(
            pipeline_state_run2.get_node_state(
                trainer_node_uid, pstate._PREVIOUS_NODE_STATES).state,
            pstate.NodeState.FAILED)
        self.assertEqual(
            pipeline_state_run2.get_node_state(transform_node_uid).state,
            pstate.NodeState.SKIPPED_PARTIAL_RUN)
        self.assertEqual(
            pipeline_state_run2.get_node_state(
                transform_node_uid, pstate._PREVIOUS_NODE_STATES).state,
            pstate.NodeState.FAILED)
        _stop_pipeline(pipeline_state_run2)
      mock_snapshot.assert_called()

  @mock.patch.object(partial_run_utils, 'snapshot')
  def test_initiate_pipeline_start_with_partial_run_default_to_nodes(
      self, mock_snapshot):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('test_pipeline', pipeline_pb2.Pipeline.SYNC)
      node_example_gen = pipeline.nodes.add().pipeline_node
      node_example_gen.node_info.id = 'ExampleGen'
      node_example_gen.downstream_nodes.extend(['Transform'])
      node_transform = pipeline.nodes.add().pipeline_node
      node_transform.node_info.id = 'Transform'
      node_transform.upstream_nodes.extend(['ExampleGen'])
      node_transform.downstream_nodes.extend(['Trainer'])
      node_trainer = pipeline.nodes.add().pipeline_node
      node_trainer.node_info.id = 'Trainer'
      node_trainer.upstream_nodes.extend(['Transform'])

      expected_pipeline = copy.deepcopy(pipeline)
      partial_run_utils.set_latest_pipeline_run_strategy(
          expected_pipeline.runtime_spec.snapshot_settings)

      expected_pipeline.nodes[
          0].pipeline_node.execution_options.skip.reuse_artifacts_mode = pipeline_pb2.NodeExecutionOptions.Skip.REQUIRED
      expected_pipeline.nodes[
          1].pipeline_node.execution_options.run.perform_snapshot = True
      expected_pipeline.nodes[
          1].pipeline_node.execution_options.run.depends_on_snapshot = True
      expected_pipeline.nodes[
          2].pipeline_node.execution_options.run.SetInParent()

      partial_run_option = pipeline_pb2.PartialRun(
          from_nodes=['Transform'],
          snapshot_settings=partial_run_utils.latest_pipeline_snapshot_settings(
          ))
      with pipeline_ops.initiate_pipeline_start(
          m, pipeline, partial_run_option=partial_run_option) as pipeline_state:
        self.assertEqual(expected_pipeline, pipeline_state.pipeline)
        mock_snapshot.assert_called_once()

  @mock.patch.object(partial_run_utils, 'snapshot')
  def test_partial_run_defaults_to_latest_pipeline_run_strategy(
      self, mock_snapshot):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('test_pipeline', pipeline_pb2.Pipeline.SYNC)
      node_example_gen = pipeline.nodes.add().pipeline_node
      node_example_gen.node_info.id = 'ExampleGen'
      node_example_gen.downstream_nodes.extend(['Transform'])
      node_transform = pipeline.nodes.add().pipeline_node
      node_transform.node_info.id = 'Transform'
      node_transform.upstream_nodes.extend(['ExampleGen'])
      node_transform.downstream_nodes.extend(['Trainer'])
      node_trainer = pipeline.nodes.add().pipeline_node
      node_trainer.node_info.id = 'Trainer'
      node_trainer.upstream_nodes.extend(['Transform'])

      # partial_run_option without artifact_reuse_strategy should default to
      # latest_pipeline_run_strategy.
      partial_run_option = pipeline_pb2.PartialRun(
          from_nodes=['Transform'], to_nodes=['Trainer'])

      expected_pipeline = copy.deepcopy(pipeline)
      partial_run_utils.set_latest_pipeline_run_strategy(
          expected_pipeline.runtime_spec.snapshot_settings)
      expected_pipeline.nodes[
          0].pipeline_node.execution_options.skip.reuse_artifacts_mode = pipeline_pb2.NodeExecutionOptions.Skip.REQUIRED
      expected_pipeline.nodes[
          1].pipeline_node.execution_options.run.perform_snapshot = True
      expected_pipeline.nodes[
          1].pipeline_node.execution_options.run.depends_on_snapshot = True
      expected_pipeline.nodes[
          2].pipeline_node.execution_options.run.SetInParent()

      with pipeline_ops.initiate_pipeline_start(
          m, pipeline, partial_run_option=partial_run_option) as pipeline_state:
        self.assertEqual(expected_pipeline, pipeline_state.pipeline)
        mock_snapshot.assert_called_once()

  @parameterized.named_parameters(
      dict(testcase_name='async', pipeline=_test_pipeline('pipeline1')),
      dict(
          testcase_name='sync',
          pipeline=_test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC)))
  def test_stop_pipeline_non_existent_or_inactive(self, pipeline):
    with self._mlmd_connection as m:
      # Stop pipeline without creating one.
      with self.assertRaises(status_lib.StatusNotOkError) as exception_context:
        pipeline_ops.stop_pipeline(m,
                                   task_lib.PipelineUid.from_pipeline(pipeline))
      self.assertEqual(status_lib.Code.NOT_FOUND,
                       exception_context.exception.code)

      # Initiate pipeline start and mark it completed.
      pipeline_ops.initiate_pipeline_start(m, pipeline)
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        pipeline_state.initiate_stop(status_lib.Status(code=status_lib.Code.OK))
        pipeline_state.set_pipeline_execution_state(
            metadata_store_pb2.Execution.COMPLETE)

      # Try to initiate stop again.
      with self.assertRaises(status_lib.StatusNotOkError) as exception_context:
        pipeline_ops.stop_pipeline(m, pipeline_uid)
      self.assertEqual(status_lib.Code.NOT_FOUND,
                       exception_context.exception.code)

  @parameterized.named_parameters(
      dict(testcase_name='async', pipeline=_test_pipeline('pipeline1')),
      dict(
          testcase_name='sync',
          pipeline=_test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC)))
  def test_stop_pipeline_wait_for_inactivation(self, pipeline):
    with self._mlmd_connection as m:
      pipeline_state = pipeline_ops.initiate_pipeline_start(m, pipeline)

      def _inactivate(pipeline_state):
        time.sleep(2.0)
        with pipeline_ops._PIPELINE_OPS_LOCK:
          with pipeline_state:
            pipeline_state.set_pipeline_execution_state(
                metadata_store_pb2.Execution.COMPLETE)

      thread = threading.Thread(target=_inactivate, args=(pipeline_state,))
      thread.start()

      pipeline_ops.stop_pipeline(
          m, task_lib.PipelineUid.from_pipeline(pipeline), timeout_secs=20.0)

      thread.join()

  @parameterized.named_parameters(
      dict(testcase_name='async', pipeline=_test_pipeline('pipeline1')),
      dict(
          testcase_name='sync',
          pipeline=_test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC)))
  def test_stop_pipeline_wait_for_inactivation_timeout(self, pipeline):
    with self._mlmd_connection as m:
      pipeline_ops.initiate_pipeline_start(m, pipeline)

      with self.assertRaisesRegex(
          status_lib.StatusNotOkError,
          'Timed out.*waiting for execution inactivation.'
      ) as exception_context:
        pipeline_ops.stop_pipeline(
            m, task_lib.PipelineUid.from_pipeline(pipeline), timeout_secs=1.0)
      self.assertEqual(status_lib.Code.DEADLINE_EXCEEDED,
                       exception_context.exception.code)

  def test_stop_node_wait_for_inactivation(self):
    pipeline = test_async_pipeline.create_pipeline()
    pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
    node_uid = task_lib.NodeUid(node_id='my_trainer', pipeline_uid=pipeline_uid)
    with self._mlmd_connection as m:
      pstate.PipelineState.new(m, pipeline)

      def _inactivate():
        time.sleep(2.0)
        with pipeline_ops._PIPELINE_OPS_LOCK:
          with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
            with pipeline_state.node_state_update_context(
                node_uid) as node_state:
              node_state.update(
                  pstate.NodeState.STOPPED,
                  status_lib.Status(code=status_lib.Code.CANCELLED))

      thread = threading.Thread(target=_inactivate, args=())
      thread.start()
      pipeline_ops.stop_node(m, node_uid, timeout_secs=20.0)
      thread.join()

      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(node_uid)
        self.assertEqual(pstate.NodeState.STOPPED, node_state.state)

      # Restart node.
      with pipeline_ops.initiate_node_start(m, node_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(node_uid)
        self.assertEqual(pstate.NodeState.STARTING, node_state.state)

  def test_stop_node_wait_for_inactivation_timeout(self):
    pipeline = test_async_pipeline.create_pipeline()
    pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
    node_uid = task_lib.NodeUid(node_id='my_trainer', pipeline_uid=pipeline_uid)
    with self._mlmd_connection as m:
      pstate.PipelineState.new(m, pipeline)
      with self.assertRaisesRegex(
          status_lib.StatusNotOkError,
          'Timed out.*waiting for node inactivation.') as exception_context:
        pipeline_ops.stop_node(m, node_uid, timeout_secs=1.0)
      self.assertEqual(status_lib.Code.DEADLINE_EXCEEDED,
                       exception_context.exception.code)

      # Even if `wait_for_inactivation` times out, the node should be in state
      # STOPPING or STOPPED to prevent future triggers.
      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(node_uid)
        self.assertIn(node_state.state,
                      (pstate.NodeState.STOPPING, pstate.NodeState.STOPPED))

  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  @mock.patch.object(async_pipeline_task_gen, 'AsyncPipelineTaskGenerator')
  def test_orchestrate_active_pipelines(self, mock_async_task_gen,
                                        mock_sync_task_gen):
    with self._mlmd_connection as m:
      # Sync and async active pipelines.
      async_pipelines = [
          _test_pipeline('pipeline1'),
          _test_pipeline('pipeline2'),
      ]
      sync_pipelines = [
          _test_pipeline('pipeline3', pipeline_pb2.Pipeline.SYNC),
          _test_pipeline('pipeline4', pipeline_pb2.Pipeline.SYNC),
      ]

      for pipeline in async_pipelines + sync_pipelines:
        pipeline_ops.initiate_pipeline_start(m, pipeline)

      # Active executions for active async pipelines.
      mock_async_task_gen.return_value.generate.side_effect = [
          [
              test_utils.create_exec_node_task(
                  task_lib.NodeUid(
                      pipeline_uid=task_lib.PipelineUid.from_pipeline(
                          async_pipelines[0]),
                      node_id='Transform'))
          ],
          [
              test_utils.create_exec_node_task(
                  task_lib.NodeUid(
                      pipeline_uid=task_lib.PipelineUid.from_pipeline(
                          async_pipelines[1]),
                      node_id='Trainer'))
          ],
      ]

      # Active executions for active sync pipelines.
      mock_sync_task_gen.return_value.generate.side_effect = [
          [
              test_utils.create_exec_node_task(
                  task_lib.NodeUid(
                      pipeline_uid=task_lib.PipelineUid.from_pipeline(
                          sync_pipelines[0]),
                      node_id='Trainer'))
          ],
          [
              test_utils.create_exec_node_task(
                  task_lib.NodeUid(
                      pipeline_uid=task_lib.PipelineUid.from_pipeline(
                          sync_pipelines[1]),
                      node_id='Validator'))
          ],
      ]

      task_queue = tq.TaskQueue()
      pipeline_ops.orchestrate(m, task_queue,
                               service_jobs.DummyServiceJobManager())

      self.assertEqual(2, mock_async_task_gen.return_value.generate.call_count)
      self.assertEqual(2, mock_sync_task_gen.return_value.generate.call_count)

      # Verify that tasks are enqueued in the expected order.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertTrue(task_lib.is_exec_node_task(task))
      self.assertEqual(
          test_utils.create_node_uid('pipeline1', 'Transform'), task.node_uid)
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertTrue(task_lib.is_exec_node_task(task))
      self.assertEqual(
          test_utils.create_node_uid('pipeline2', 'Trainer'), task.node_uid)
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertTrue(task_lib.is_exec_node_task(task))
      self.assertEqual(
          test_utils.create_node_uid('pipeline3', 'Trainer'), task.node_uid)
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertTrue(task_lib.is_exec_node_task(task))
      self.assertEqual(
          test_utils.create_node_uid('pipeline4', 'Validator'), task.node_uid)
      self.assertTrue(task_queue.is_empty())

  @parameterized.parameters(
      _test_pipeline('pipeline1'),
      _test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC))
  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  @mock.patch.object(async_pipeline_task_gen, 'AsyncPipelineTaskGenerator')
  @mock.patch.object(task_gen_utils, 'generate_task_from_active_execution')
  def test_orchestrate_stop_initiated_pipelines(self, pipeline,
                                                mock_gen_task_from_active,
                                                mock_async_task_gen,
                                                mock_sync_task_gen):
    with self._mlmd_connection as m:
      pipeline.nodes.add().pipeline_node.node_info.id = 'ExampleGen'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Transform'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Trainer'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Evaluator'

      pipeline_ops.initiate_pipeline_start(m, pipeline)
      with pstate.PipelineState.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline)) as pipeline_state:
        pipeline_state.initiate_stop(
            status_lib.Status(code=status_lib.Code.CANCELLED))
        pipeline_execution_id = pipeline_state.execution_id

      task_queue = tq.TaskQueue()

      # For the stop-initiated pipeline, "Transform" execution task is in queue,
      # "Trainer" has an active execution in MLMD but no task in queue,
      # "Evaluator" has no active execution.
      task_queue.enqueue(
          test_utils.create_exec_node_task(
              task_lib.NodeUid(
                  pipeline_uid=task_lib.PipelineUid.from_pipeline(pipeline),
                  node_id='Transform')))
      transform_task = task_queue.dequeue()  # simulates task being processed
      mock_gen_task_from_active.side_effect = [
          test_utils.create_exec_node_task(
              node_uid=task_lib.NodeUid(
                  pipeline_uid=task_lib.PipelineUid.from_pipeline(pipeline),
                  node_id='Trainer'),
              is_cancelled=True), None, None, None, None
      ]

      pipeline_ops.orchestrate(m, task_queue, self._mock_service_job_manager)

      # There are no active pipelines so these shouldn't be called.
      mock_async_task_gen.assert_not_called()
      mock_sync_task_gen.assert_not_called()

      # stop_node_services should be called for ExampleGen which is a pure
      # service node.
      self._mock_service_job_manager.stop_node_services.assert_called_once_with(
          mock.ANY, 'ExampleGen')
      self._mock_service_job_manager.reset_mock()

      task_queue.task_done(transform_task)  # Pop out transform task.

      # CancelNodeTask for the "Transform" ExecNodeTask should be next.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertTrue(task_lib.is_cancel_node_task(task))
      self.assertEqual('Transform', task.node_uid.node_id)

      # ExecNodeTask (with is_cancelled=True) for "Trainer" is next.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertTrue(task_lib.is_exec_node_task(task))
      self.assertEqual('Trainer', task.node_uid.node_id)
      self.assertTrue(task.is_cancelled)

      self.assertTrue(task_queue.is_empty())

      mock_gen_task_from_active.assert_has_calls([
          mock.call(
              m,
              pipeline_state.pipeline,
              pipeline.nodes[2].pipeline_node,
              mock.ANY,
              is_cancelled=True),
          mock.call(
              m,
              pipeline_state.pipeline,
              pipeline.nodes[3].pipeline_node,
              mock.ANY,
              is_cancelled=True)
      ])
      self.assertEqual(2, mock_gen_task_from_active.call_count)

      # Pipeline execution should continue to be active since active node
      # executions were found in the last call to `orchestrate`.
      [execution] = m.store.get_executions_by_id([pipeline_execution_id])
      self.assertTrue(execution_lib.is_execution_active(execution))

      # Call `orchestrate` again; this time there are no more active node
      # executions so the pipeline should be marked as cancelled.
      pipeline_ops.orchestrate(m, task_queue, self._mock_service_job_manager)
      self.assertTrue(task_queue.is_empty())
      [execution] = m.store.get_executions_by_id([pipeline_execution_id])
      self.assertEqual(metadata_store_pb2.Execution.CANCELED,
                       execution.last_known_state)

      # stop_node_services should be called on Transform which is a mixed
      # service node.
      self._mock_service_job_manager.stop_node_services.assert_has_calls(
          [mock.call(mock.ANY, 'Transform')])

      # Check that all the node states are STOPPED.
      node_states_dict = pstate._get_node_states_dict(execution)
      self.assertLen(node_states_dict, 4)
      self.assertSetEqual(
          set([pstate.NodeState.STOPPED]),
          set(n.state for n in node_states_dict.values()))

  @parameterized.parameters(
      _test_pipeline('pipeline1'),
      _test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC))
  def test_orchestrate_update_initiated_pipelines(self, pipeline):
    with self._mlmd_connection as m:
      pipeline.nodes.add().pipeline_node.node_info.id = 'ExampleGen'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Transform'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Trainer'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Evaluator'

      pipeline_ops.initiate_pipeline_start(m, pipeline)

      task_queue = tq.TaskQueue()

      for node_id in ('Transform', 'Trainer', 'Evaluator'):
        task_queue.enqueue(
            test_utils.create_exec_node_task(
                task_lib.NodeUid(
                    pipeline_uid=task_lib.PipelineUid.from_pipeline(pipeline),
                    node_id=node_id)))
      pipeline_state = pipeline_ops._initiate_pipeline_update(
          m,
          pipeline,
          update_options=pipeline_pb2.UpdateOptions(
              reload_policy=pipeline_pb2.UpdateOptions.ALL))
      with pipeline_state:
        self.assertTrue(pipeline_state.is_update_initiated())

      pipeline_ops.orchestrate(m, task_queue, self._mock_service_job_manager)

      # stop_node_services should be called for ExampleGen which is a pure
      # service node.
      self._mock_service_job_manager.stop_node_services.assert_called_once_with(
          mock.ANY, 'ExampleGen')
      self._mock_service_job_manager.reset_mock()

      # Simulate completion of all the exec node tasks.
      for node_id in ('Transform', 'Trainer', 'Evaluator'):
        task = task_queue.dequeue()
        task_queue.task_done(task)
        self.assertTrue(task_lib.is_exec_node_task(task))
        self.assertEqual(node_id, task.node_uid.node_id)

      # Verify that cancellation tasks were enqueued in the last `orchestrate`
      # call, and dequeue them.
      for node_id in ('Transform', 'Trainer', 'Evaluator'):
        task = task_queue.dequeue()
        task_queue.task_done(task)
        self.assertTrue(task_lib.is_cancel_node_task(task))
        self.assertEqual(node_id, task.node_uid.node_id)
        self.assertTrue(task.pause)

      self.assertTrue(task_queue.is_empty())

      # Pipeline continues to be in update initiated state until all
      # ExecNodeTasks have been dequeued (which was not the case when last
      # `orchestrate` call was made).
      with pipeline_state:
        self.assertTrue(pipeline_state.is_update_initiated())

      pipeline_ops.orchestrate(m, task_queue, self._mock_service_job_manager)

      # stop_node_services should be called for Transform (mixed service node)
      # too since corresponding ExecNodeTask has been processed.
      self._mock_service_job_manager.stop_node_services.assert_has_calls(
          [mock.call(mock.ANY, 'Transform')])

      # Check that the node states are STARTED.
      [execution] = m.store.get_executions_by_id([pipeline_state.execution_id])
      node_states_dict = pstate._get_node_states_dict(execution)
      self.assertLen(node_states_dict, 4)
      self.assertSetEqual(
          set([pstate.NodeState.STARTED]),
          set(n.state for n in node_states_dict.values()))

      # Pipeline should no longer be in update-initiated state but be active.
      with pipeline_state:
        self.assertFalse(pipeline_state.is_update_initiated())
        self.assertTrue(pipeline_state.is_active())

  def test_orchestrate_update_initiated_pipelines_options(self):
    pipeline = _test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC)
    with self._mlmd_connection as m:
      pipeline.nodes.add().pipeline_node.node_info.id = 'ExampleGen'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Transform'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Trainer'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Evaluator'

      pipeline_ops.initiate_pipeline_start(m, pipeline)

      task_queue = tq.TaskQueue()

      for node_id in ('Transform', 'Trainer', 'Evaluator'):
        task_queue.enqueue(
            test_utils.create_exec_node_task(
                task_lib.NodeUid(
                    pipeline_uid=task_lib.PipelineUid.from_pipeline(pipeline),
                    node_id=node_id)))
      pipeline_state = pipeline_ops._initiate_pipeline_update(
          m,
          pipeline,
          update_options=pipeline_pb2.UpdateOptions(
              reload_policy=pipeline_pb2.UpdateOptions.PARTIAL,
              reload_nodes=['Transform', 'Trainer']))
      with pipeline_state:
        self.assertTrue(pipeline_state.is_update_initiated())

      pipeline_ops.orchestrate(m, task_queue, self._mock_service_job_manager)

      # stop_node_services should not be called for ExampleGen since it is not
      # reloaded according to the options.
      self._mock_service_job_manager.stop_node_services.assert_not_called()

      # Simulate completion of all the exec node tasks except evaluator.
      for node_id in ('Transform', 'Trainer', 'Evaluator'):
        task = task_queue.dequeue()
        task_queue.task_done(task)
        self.assertTrue(task_lib.is_exec_node_task(task))
        self.assertEqual(node_id, task.node_uid.node_id)

      # Verify that cancellation tasks were enqueued in the last `orchestrate`
      # call, and dequeue them.
      for node_id in ('Transform', 'Trainer'):
        task = task_queue.dequeue()
        task_queue.task_done(task)
        self.assertTrue(task_lib.is_cancel_node_task(task))
        self.assertEqual(node_id, task.node_uid.node_id)
        self.assertTrue(task.pause)

      # Pipeline continues to be in update initiated state until all
      # ExecNodeTasks have been dequeued (which was not the case when last
      # `orchestrate` call was made).
      with pipeline_state:
        self.assertTrue(pipeline_state.is_update_initiated())

      pipeline_ops.orchestrate(m, task_queue, self._mock_service_job_manager)

      # stop_node_services should be called for Transform (mixed service node)
      # too since corresponding ExecNodeTask has been processed.
      self._mock_service_job_manager.stop_node_services.assert_has_calls(
          [mock.call(mock.ANY, 'Transform')])

      # Pipeline should no longer be in update-initiated state but be active.
      with pipeline_state:
        self.assertFalse(pipeline_state.is_update_initiated())
        self.assertTrue(pipeline_state.is_active())

      self.assertTrue(task_queue.is_empty())

  def test_update_pipeline_waits_for_update_application(self):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('pipeline1')
      pipeline_state = pipeline_ops.initiate_pipeline_start(m, pipeline)

      def _apply_update(pipeline_state):
        # Wait for the pipeline to be in update initiated state.
        while True:
          with pipeline_state:
            if pipeline_state.is_update_initiated():
              break
          time.sleep(0.5)
        # Now apply the update.
        with pipeline_ops._PIPELINE_OPS_LOCK:
          with pipeline_state:
            pipeline_state.apply_pipeline_update()

      thread = threading.Thread(target=_apply_update, args=(pipeline_state,))
      thread.start()
      pipeline_ops.update_pipeline(
          m,
          pipeline,
          update_options=pipeline_pb2.UpdateOptions(
              reload_policy=pipeline_pb2.UpdateOptions.ALL),
          timeout_secs=10.0)
      thread.join()

  def test_update_pipeline_wait_for_update_timeout(self):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('pipeline1')
      pipeline_ops.initiate_pipeline_start(m, pipeline)
      with self.assertRaisesRegex(status_lib.StatusNotOkError,
                                  'Timed out.*waiting for pipeline update'):

        pipeline_ops.update_pipeline(
            m,
            pipeline,
            update_options=pipeline_pb2.UpdateOptions(
                reload_policy=pipeline_pb2.UpdateOptions.ALL),
            timeout_secs=3.0)

  @parameterized.parameters(
      _test_pipeline('pipeline1'),
      _test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC))
  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  @mock.patch.object(async_pipeline_task_gen, 'AsyncPipelineTaskGenerator')
  @mock.patch.object(task_gen_utils, 'generate_task_from_active_execution')
  def test_active_pipelines_with_stopped_nodes(self, pipeline,
                                               mock_gen_task_from_active,
                                               mock_async_task_gen,
                                               mock_sync_task_gen):
    if pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC:
      mock_task_gen = mock_sync_task_gen
    else:
      mock_task_gen = mock_async_task_gen

    with self._mlmd_connection as m:
      pipeline.nodes.add().pipeline_node.node_info.id = 'ExampleGen'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Transform'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Trainer'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Evaluator'

      example_gen_node_uid = task_lib.NodeUid.from_pipeline_node(
          pipeline, pipeline.nodes[0].pipeline_node)

      transform_node_uid = task_lib.NodeUid.from_pipeline_node(
          pipeline, pipeline.nodes[1].pipeline_node)
      transform_task = test_utils.create_exec_node_task(
          node_uid=transform_node_uid)

      trainer_node_uid = task_lib.NodeUid.from_pipeline_node(
          pipeline, pipeline.nodes[2].pipeline_node)
      trainer_task = test_utils.create_exec_node_task(node_uid=trainer_node_uid)

      evaluator_node_uid = task_lib.NodeUid.from_pipeline_node(
          pipeline, pipeline.nodes[3].pipeline_node)
      evaluator_task = test_utils.create_exec_node_task(
          node_uid=evaluator_node_uid)
      cancelled_evaluator_task = test_utils.create_exec_node_task(
          node_uid=evaluator_node_uid, is_cancelled=True)

      pipeline_ops.initiate_pipeline_start(m, pipeline)
      with pstate.PipelineState.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline)) as pipeline_state:
        # Stop example-gen, trainer and evaluator.
        with pipeline_state.node_state_update_context(
            example_gen_node_uid) as node_state:
          node_state.update(pstate.NodeState.STOPPING,
                            status_lib.Status(code=status_lib.Code.CANCELLED))
        with pipeline_state.node_state_update_context(
            trainer_node_uid) as node_state:
          node_state.update(pstate.NodeState.STOPPING,
                            status_lib.Status(code=status_lib.Code.CANCELLED))
        with pipeline_state.node_state_update_context(
            evaluator_node_uid) as node_state:
          node_state.update(pstate.NodeState.STOPPING,
                            status_lib.Status(code=status_lib.Code.ABORTED))

      task_queue = tq.TaskQueue()

      # Simulate a new transform execution being triggered.
      mock_task_gen.return_value.generate.return_value = [transform_task]
      # Simulate ExecNodeTask for trainer already present in the task queue.
      task_queue.enqueue(trainer_task)
      # Simulate Evaluator having an active execution in MLMD.
      mock_gen_task_from_active.side_effect = [evaluator_task]

      pipeline_ops.orchestrate(m, task_queue, self._mock_service_job_manager)
      self.assertEqual(1, mock_task_gen.return_value.generate.call_count)

      # stop_node_services should be called on example-gen which is a pure
      # service node.
      self._mock_service_job_manager.stop_node_services.assert_called_once_with(
          mock.ANY, 'ExampleGen')

      # Verify that tasks are enqueued in the expected order:

      # Pre-existing trainer task.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertEqual(trainer_task, task)

      # CancelNodeTask for trainer.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertTrue(task_lib.is_cancel_node_task(task))
      self.assertEqual(trainer_node_uid, task.node_uid)

      # ExecNodeTask with is_cancelled=True for evaluator.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertTrue(cancelled_evaluator_task, task)

      # ExecNodeTask for newly triggered transform node.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertEqual(transform_task, task)

      # No more tasks.
      self.assertTrue(task_queue.is_empty())

  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  def test_handling_finalize_pipeline_task(self, task_gen):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC)
      pipeline_ops.initiate_pipeline_start(m, pipeline)
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      finalize_reason = status_lib.Status(
          code=status_lib.Code.ABORTED, message='foo bar')
      task_gen.return_value.generate.side_effect = [
          [
              task_lib.FinalizePipelineTask(
                  pipeline_uid=pipeline_uid, status=finalize_reason)
          ],
      ]

      task_queue = tq.TaskQueue()
      pipeline_ops.orchestrate(m, task_queue,
                               service_jobs.DummyServiceJobManager())
      task_gen.return_value.generate.assert_called_once()
      self.assertTrue(task_queue.is_empty())

      # Load pipeline state and verify stop initiation.
      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        self.assertEqual(finalize_reason,
                         pipeline_state.stop_initiated_reason())

  @mock.patch.object(async_pipeline_task_gen, 'AsyncPipelineTaskGenerator')
  def test_handling_finalize_node_task(self, task_gen):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('pipeline1')
      pipeline.nodes.add().pipeline_node.node_info.id = 'Transform'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Trainer'
      pipeline_ops.initiate_pipeline_start(m, pipeline)
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      transform_node_uid = task_lib.NodeUid(
          pipeline_uid=pipeline_uid, node_id='Transform')
      trainer_node_uid = task_lib.NodeUid(
          pipeline_uid=pipeline_uid, node_id='Trainer')
      task_gen.return_value.generate.side_effect = [
          [
              test_utils.create_exec_node_task(transform_node_uid),
              task_lib.UpdateNodeStateTask(
                  node_uid=trainer_node_uid, state=pstate.NodeState.FAILED),
          ],
      ]

      task_queue = tq.TaskQueue()
      pipeline_ops.orchestrate(m, task_queue,
                               service_jobs.DummyServiceJobManager())
      task_gen.return_value.generate.assert_called_once()
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertTrue(task_lib.is_exec_node_task(task))
      self.assertEqual(transform_node_uid, task.node_uid)

      # Load pipeline state and verify trainer node state.
      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(trainer_node_uid)
        self.assertEqual(pstate.NodeState.FAILED, node_state.state)

  def test_to_status_not_ok_error_decorator(self):

    @pipeline_ops._to_status_not_ok_error
    def fn1():
      raise RuntimeError('test error 1')

    @pipeline_ops._to_status_not_ok_error
    def fn2():
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.ALREADY_EXISTS, message='test error 2')

    with self.assertRaisesRegex(status_lib.StatusNotOkError,
                                'test error 1') as ctxt:
      fn1()
    self.assertEqual(status_lib.Code.UNKNOWN, ctxt.exception.code)

    with self.assertRaisesRegex(status_lib.StatusNotOkError,
                                'test error 2') as ctxt:
      fn2()
    self.assertEqual(status_lib.Code.ALREADY_EXISTS, ctxt.exception.code)

  @parameterized.parameters(
      _test_pipeline('pipeline1'),
      _test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC))
  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  @mock.patch.object(async_pipeline_task_gen, 'AsyncPipelineTaskGenerator')
  def test_executor_node_stop_then_start_flow(self, pipeline,
                                              mock_async_task_gen,
                                              mock_sync_task_gen):
    service_job_manager = service_jobs.DummyServiceJobManager()
    with self._mlmd_connection as m:
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      pipeline.nodes.add().pipeline_node.node_info.id = 'Trainer'
      trainer_node_uid = task_lib.NodeUid.from_pipeline_node(
          pipeline, pipeline.nodes[0].pipeline_node)

      # Start pipeline and stop trainer.
      pipeline_ops.initiate_pipeline_start(m, pipeline)
      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        with pipeline_state.node_state_update_context(
            trainer_node_uid) as node_state:
          node_state.update(pstate.NodeState.STOPPING,
                            status_lib.Status(code=status_lib.Code.CANCELLED))

      task_queue = tq.TaskQueue()

      # Simulate ExecNodeTask for trainer already present in the task queue.
      trainer_task = test_utils.create_exec_node_task(node_uid=trainer_node_uid)
      task_queue.enqueue(trainer_task)

      pipeline_ops.orchestrate(m, task_queue, service_job_manager)

      # Dequeue pre-existing trainer task.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertEqual(trainer_task, task)

      # Dequeue CancelNodeTask for trainer.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertTrue(task_lib.is_cancel_node_task(task))
      self.assertEqual(trainer_node_uid, task.node_uid)

      self.assertTrue(task_queue.is_empty())

      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(trainer_node_uid)
        self.assertEqual(pstate.NodeState.STOPPING, node_state.state)
        self.assertEqual(status_lib.Code.CANCELLED, node_state.status.code)

      pipeline_ops.orchestrate(m, task_queue, service_job_manager)

      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(trainer_node_uid)
        self.assertEqual(pstate.NodeState.STOPPED, node_state.state)
        self.assertEqual(status_lib.Code.CANCELLED, node_state.status.code)

      pipeline_ops.initiate_node_start(m, trainer_node_uid)
      pipeline_ops.orchestrate(m, task_queue, service_job_manager)

      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(trainer_node_uid)
        self.assertEqual(pstate.NodeState.STARTED, node_state.state)

  @parameterized.parameters(
      _test_pipeline('pipeline1'),
      _test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC))
  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  @mock.patch.object(async_pipeline_task_gen, 'AsyncPipelineTaskGenerator')
  def test_pure_service_node_stop_then_start_flow(self, pipeline,
                                                  mock_async_task_gen,
                                                  mock_sync_task_gen):
    with self._mlmd_connection as m:
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      pipeline.nodes.add().pipeline_node.node_info.id = 'ExampleGen'

      example_gen_node_uid = task_lib.NodeUid.from_pipeline_node(
          pipeline, pipeline.nodes[0].pipeline_node)

      pipeline_ops.initiate_pipeline_start(m, pipeline)
      with pstate.PipelineState.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline)) as pipeline_state:
        with pipeline_state.node_state_update_context(
            example_gen_node_uid) as node_state:
          node_state.update(pstate.NodeState.STOPPING,
                            status_lib.Status(code=status_lib.Code.CANCELLED))

      task_queue = tq.TaskQueue()

      pipeline_ops.orchestrate(m, task_queue, self._mock_service_job_manager)

      # stop_node_services should be called for ExampleGen which is a pure
      # service node.
      self._mock_service_job_manager.stop_node_services.assert_called_once_with(
          mock.ANY, 'ExampleGen')

      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(example_gen_node_uid)
        self.assertEqual(pstate.NodeState.STOPPED, node_state.state)
        self.assertEqual(status_lib.Code.CANCELLED, node_state.status.code)

      pipeline_ops.initiate_node_start(m, example_gen_node_uid)
      pipeline_ops.orchestrate(m, task_queue, self._mock_service_job_manager)

      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(example_gen_node_uid)
        self.assertEqual(pstate.NodeState.STARTED, node_state.state)

  @parameterized.parameters(
      _test_pipeline('pipeline1'),
      _test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC))
  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  @mock.patch.object(async_pipeline_task_gen, 'AsyncPipelineTaskGenerator')
  def test_mixed_service_node_stop_then_start_flow(self, pipeline,
                                                   mock_async_task_gen,
                                                   mock_sync_task_gen):
    with self._mlmd_connection as m:
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      pipeline.nodes.add().pipeline_node.node_info.id = 'Transform'

      transform_node_uid = task_lib.NodeUid.from_pipeline_node(
          pipeline, pipeline.nodes[0].pipeline_node)

      pipeline_ops.initiate_pipeline_start(m, pipeline)
      with pstate.PipelineState.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline)) as pipeline_state:
        # Stop Transform.
        with pipeline_state.node_state_update_context(
            transform_node_uid) as node_state:
          node_state.update(pstate.NodeState.STOPPING,
                            status_lib.Status(code=status_lib.Code.CANCELLED))

      task_queue = tq.TaskQueue()

      # Simulate ExecNodeTask for Transform already present in the task queue.
      transform_task = test_utils.create_exec_node_task(
          node_uid=transform_node_uid)
      task_queue.enqueue(transform_task)

      pipeline_ops.orchestrate(m, task_queue, self._mock_service_job_manager)

      # stop_node_services should not be called as there was an active
      # ExecNodeTask for Transform which is a mixed service node.
      self._mock_service_job_manager.stop_node_services.assert_not_called()

      # Dequeue pre-existing transform task.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertEqual(transform_task, task)

      # Dequeue CancelNodeTask for transform.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertTrue(task_lib.is_cancel_node_task(task))
      self.assertEqual(transform_node_uid, task.node_uid)

      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(transform_node_uid)
        self.assertEqual(pstate.NodeState.STOPPING, node_state.state)
        self.assertEqual(status_lib.Code.CANCELLED, node_state.status.code)

      pipeline_ops.orchestrate(m, task_queue, self._mock_service_job_manager)

      # stop_node_services should be called for Transform which is a mixed
      # service node and corresponding ExecNodeTask has been dequeued.
      self._mock_service_job_manager.stop_node_services.assert_called_once_with(
          mock.ANY, 'Transform')

      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(transform_node_uid)
        self.assertEqual(pstate.NodeState.STOPPED, node_state.state)
        self.assertEqual(status_lib.Code.CANCELLED, node_state.status.code)

      pipeline_ops.initiate_node_start(m, transform_node_uid)
      pipeline_ops.orchestrate(m, task_queue, self._mock_service_job_manager)

      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(transform_node_uid)
        self.assertEqual(pstate.NodeState.STARTED, node_state.state)

  @mock.patch.object(time, 'sleep')
  def test_wait_for_predicate_timeout_secs_None(self, mock_sleep):
    predicate_fn = mock.Mock()
    predicate_fn.side_effect = [False, False, False, True]
    pipeline_ops._wait_for_predicate(predicate_fn, 'testing', None)
    self.assertEqual(predicate_fn.call_count, 4)
    self.assertEqual(mock_sleep.call_count, 3)
    predicate_fn.reset_mock()
    mock_sleep.reset_mock()

    predicate_fn.side_effect = [False, False, ValueError('test error')]
    with self.assertRaisesRegex(ValueError, 'test error'):
      pipeline_ops._wait_for_predicate(predicate_fn, 'testing', None)
    self.assertEqual(predicate_fn.call_count, 3)
    self.assertEqual(mock_sleep.call_count, 2)

  def test_resume_manual_node(self):
    pipeline = test_manual_node.create_pipeline()
    runtime_parameter_utils.substitute_runtime_parameter(
        pipeline, {
            constants.PIPELINE_RUN_ID_PARAMETER_NAME: 'test-pipeline-run',
        })
    manual_node = pipeline.nodes[0].pipeline_node
    with self._mlmd_connection as m:
      pstate.PipelineState.new(m, pipeline)
      contexts = context_lib.prepare_contexts(m, manual_node.contexts)
      execution = execution_publish_utils.register_execution(
          m, manual_node.node_info.type, contexts)

      with mlmd_state.mlmd_execution_atomic_op(
          mlmd_handle=m, execution_id=execution.id) as execution:
        node_state_mlmd_value = execution.custom_properties.get(
            manual_task_scheduler.NODE_STATE_PROPERTY_KEY)
        node_state = manual_task_scheduler.ManualNodeState.from_mlmd_value(
            node_state_mlmd_value)
      self.assertEqual(node_state.state,
                       manual_task_scheduler.ManualNodeState.WAITING)

      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      node_uid = task_lib.NodeUid(
          node_id=manual_node.node_info.id, pipeline_uid=pipeline_uid)

      pipeline_ops.resume_manual_node(m, node_uid)

      with mlmd_state.mlmd_execution_atomic_op(
          mlmd_handle=m, execution_id=execution.id) as execution:
        node_state_mlmd_value = execution.custom_properties.get(
            manual_task_scheduler.NODE_STATE_PROPERTY_KEY)
        node_state = manual_task_scheduler.ManualNodeState.from_mlmd_value(
            node_state_mlmd_value)
      self.assertEqual(node_state.state,
                       manual_task_scheduler.ManualNodeState.COMPLETED)

  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  def test_update_node_state_tasks_handling(self, mock_sync_task_gen):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline(
          'pipeline1', execution_mode=pipeline_pb2.Pipeline.SYNC)
      pipeline.nodes.add().pipeline_node.node_info.id = 'ExampleGen'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Transform'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Trainer'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Evaluator'
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      eg_node_uid = task_lib.NodeUid(pipeline_uid, 'ExampleGen')
      transform_node_uid = task_lib.NodeUid(pipeline_uid, 'Transform')
      trainer_node_uid = task_lib.NodeUid(pipeline_uid, 'Trainer')
      evaluator_node_uid = task_lib.NodeUid(pipeline_uid, 'Evaluator')

      with pipeline_ops.initiate_pipeline_start(m, pipeline) as pipeline_state:
        # Set initial states for the nodes.
        with pipeline_state.node_state_update_context(
            eg_node_uid) as node_state:
          node_state.update(pstate.NodeState.RUNNING)
        with pipeline_state.node_state_update_context(
            transform_node_uid) as node_state:
          node_state.update(pstate.NodeState.STARTING)
        with pipeline_state.node_state_update_context(
            trainer_node_uid) as node_state:
          node_state.update(pstate.NodeState.STARTED)
        with pipeline_state.node_state_update_context(
            evaluator_node_uid) as node_state:
          node_state.update(pstate.NodeState.RUNNING)

      mock_sync_task_gen.return_value.generate.side_effect = [
          [
              task_lib.UpdateNodeStateTask(
                  node_uid=eg_node_uid, state=pstate.NodeState.COMPLETE),
              task_lib.UpdateNodeStateTask(
                  node_uid=trainer_node_uid, state=pstate.NodeState.RUNNING),
              task_lib.UpdateNodeStateTask(
                  node_uid=evaluator_node_uid,
                  state=pstate.NodeState.FAILED,
                  status=status_lib.Status(
                      code=status_lib.Code.ABORTED, message='foobar error'))
          ],
      ]

      task_queue = tq.TaskQueue()
      pipeline_ops.orchestrate(m, task_queue,
                               service_jobs.DummyServiceJobManager())
      self.assertEqual(1, mock_sync_task_gen.return_value.generate.call_count)

      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        self.assertEqual(pstate.NodeState.COMPLETE,
                         pipeline_state.get_node_state(eg_node_uid).state)
        self.assertEqual(
            pstate.NodeState.STARTED,
            pipeline_state.get_node_state(transform_node_uid).state)
        self.assertEqual(pstate.NodeState.RUNNING,
                         pipeline_state.get_node_state(trainer_node_uid).state)
        self.assertEqual(
            pstate.NodeState.FAILED,
            pipeline_state.get_node_state(evaluator_node_uid).state)
        self.assertEqual(
            status_lib.Status(
                code=status_lib.Code.ABORTED, message='foobar error'),
            pipeline_state.get_node_state(evaluator_node_uid).status)

  @parameterized.parameters(
      _test_pipeline('pipeline1'),
      _test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC))
  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  @mock.patch.object(async_pipeline_task_gen, 'AsyncPipelineTaskGenerator')
  def test_stop_node_services_failure(self, pipeline, mock_async_task_gen,
                                      mock_sync_task_gen):
    with self._mlmd_connection as m:
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      pipeline.nodes.add().pipeline_node.node_info.id = 'ExampleGen'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Transform'

      example_gen_node_uid = task_lib.NodeUid.from_pipeline_node(
          pipeline, pipeline.nodes[0].pipeline_node)
      transform_node_uid = task_lib.NodeUid.from_pipeline_node(
          pipeline, pipeline.nodes[1].pipeline_node)

      pipeline_ops.initiate_pipeline_start(m, pipeline)
      with pstate.PipelineState.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline)) as pipeline_state:
        with pipeline_state.node_state_update_context(
            example_gen_node_uid) as node_state:
          node_state.update(pstate.NodeState.STOPPING,
                            status_lib.Status(code=status_lib.Code.CANCELLED))
        with pipeline_state.node_state_update_context(
            transform_node_uid) as node_state:
          node_state.update(pstate.NodeState.STOPPING,
                            status_lib.Status(code=status_lib.Code.CANCELLED))

      task_queue = tq.TaskQueue()

      # Simulate failure of stop_node_services.
      self._mock_service_job_manager.stop_node_services.return_value = False

      pipeline_ops.orchestrate(m, task_queue, self._mock_service_job_manager)

      self._mock_service_job_manager.stop_node_services.assert_has_calls(
          [mock.call(mock.ANY, 'ExampleGen'),
           mock.call(mock.ANY, 'Transform')],
          any_order=True)

      # Node state should be STOPPING, not STOPPED since stop_node_services
      # failed.
      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(example_gen_node_uid)
        self.assertEqual(pstate.NodeState.STOPPING, node_state.state)
        node_state = pipeline_state.get_node_state(transform_node_uid)
        self.assertEqual(pstate.NodeState.STOPPING, node_state.state)

  def test_pipeline_run_deadline_exceeded(self):

    class _TestEnv(env.Env):
      """TestEnv returns orchestration_options with 1 sec deadline."""

      def get_orchestration_options(self, pipeline):
        return orchestration_options.OrchestrationOptions(deadline_secs=1)

    with _TestEnv():
      with self._mlmd_connection as m:
        pipeline = _test_pipeline('pipeline', pipeline_pb2.Pipeline.SYNC)
        pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
        pipeline_ops.initiate_pipeline_start(m, pipeline)
        time.sleep(3)  # To trigger the deadline.
        pipeline_ops.orchestrate(m, tq.TaskQueue(),
                                 self._mock_service_job_manager)
        with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
          self.assertTrue(pipeline_state.is_stop_initiated())
          status = pipeline_state.stop_initiated_reason()
          self.assertEqual(status_lib.Code.DEADLINE_EXCEEDED, status.code)
          self.assertEqual(
              'Pipeline aborted due to exceeding deadline (1 secs)',
              status.message)

  @mock.patch.object(outputs_utils, 'OutputsResolver')
  def test_external_executions_end_to_end(self, outputs_resolver):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline_with_example_gen('test_pipeline')
      node_uid = task_lib.NodeUid.from_pipeline_node(
          pipeline, pipeline.nodes[0].pipeline_node)
      pipeline_state = pipeline_ops.initiate_pipeline_start(m, pipeline)
      pipeline_run_id = pipeline_state.pipeline_run_id

      # No executions are registered yet, so get_external_executions returns
      # an empty result.
      empty_get_result = pipeline_ops.get_external_executions(
          m, node_uid, pipeline_run_id)
      self.assertEmpty(empty_get_result)

      # Register two executions; register_external_executions should return the
      # two executions
      register_result = pipeline_ops.register_external_executions(
          m, node_uid, pipeline_state.pipeline_run_id, [{
              'test_property': 'test_value_0'
          }, {
              'test_property': 'test_value_1'
          }])
      self.assertLen(register_result, 2)
      self.assertSameElements(
          {
              'test_property':
                  metadata_store_pb2.Value(string_value='test_value_0')
          }, register_result[0].custom_properties)
      self.assertSameElements(
          {
              'test_property':
                  metadata_store_pb2.Value(string_value='test_value_1')
          }, register_result[1].custom_properties)
      self.assertEqual(metadata_store_pb2.Execution.State.NEW,
                       register_result[0].last_known_state)
      self.assertEqual(metadata_store_pb2.Execution.State.NEW,
                       register_result[1].last_known_state)

      # get_external_executions should return the two registered executions.
      get_result = pipeline_ops.get_external_executions(m, node_uid,
                                                        pipeline_run_id)
      self.assertLen(get_result, 2)
      ignored_fields = [
          'create_time_since_epoch', 'last_update_time_since_epoch'
      ]
      self.assertProtoPartiallyEquals(get_result[0], register_result[0],
                                      ignored_fields)
      self.assertProtoPartiallyEquals(get_result[1], register_result[1],
                                      ignored_fields)

      execution_id = get_result[0].id
      uri = '/test/0/out'
      stateful_working_dir = '/test/0/stateful_working_dir'
      tmp_dir = '/test/0/tmp'
      outputs_resolver.return_value.get_executor_output_uri.return_value = ''
      outputs_resolver.return_value.get_stateful_working_directory.return_value = stateful_working_dir
      outputs_resolver.return_value.make_tmp_dir.return_value = tmp_dir
      output_artifact = types.Artifact(
          metadata_store_pb2.ArtifactType(name='ExamplesPath'))
      output_artifact.uri = uri
      outputs_resolver.return_value.generate_output_artifacts.return_value = {
          'examples': [output_artifact]
      }

      # Start the first execution.
      execution_invocation = pipeline_ops.start_external_execution(
          m, node_uid, pipeline_run_id, execution_id)
      self.assertEqual(tmp_dir, execution_invocation.tmp_dir)
      self.assertEqual(stateful_working_dir,
                       execution_invocation.stateful_working_dir)
      self.assertEqual(
          metadata_store_service_pb2.ArtifactStructList(elements=[
              metadata_store_service_pb2.ArtifactStruct(
                  artifact=(metadata_store_service_pb2.ArtifactAndType(
                      artifact=metadata_store_pb2.Artifact(uri=uri),
                      type=metadata_store_pb2.ArtifactType(
                          name='ExamplesPath'))))
          ]), execution_invocation.output_dict['examples'])

      # get_external_executions should return the two registered executions.
      # Only the first execution should be running.
      get_after_start_result = pipeline_ops.get_external_executions(
          m, node_uid, pipeline_run_id)
      ignored_fields_with_state = [
          'create_time_since_epoch', 'last_update_time_since_epoch',
          'last_known_state'
      ]
      self.assertProtoPartiallyEquals(get_after_start_result[0],
                                      register_result[0],
                                      ignored_fields_with_state)
      self.assertProtoPartiallyEquals(get_after_start_result[1],
                                      register_result[1],
                                      ignored_fields_with_state)
      self.assertEqual(metadata_store_pb2.Execution.State.RUNNING,
                       get_after_start_result[0].last_known_state)
      self.assertEqual(metadata_store_pb2.Execution.State.NEW,
                       get_after_start_result[1].last_known_state)

      # Finish the first execution.
      pipeline_ops.finish_external_execution(
          m, node_uid, pipeline_run_id, execution_id,
          execution_result_pb2.ExecutorOutput(
              output_artifacts={
                  'examples':
                      execution_result_pb2.ExecutorOutput.ArtifactList(
                          artifacts=[
                              metadata_store_pb2.Artifact(
                                  type='ExamplesPath', uri=uri)
                          ])
              }))

      # get_external_executions should return the two registered executions.
      # Only the first execution should be completed.
      get_after_finish_result = pipeline_ops.get_external_executions(
          m, node_uid, pipeline_run_id)
      self.assertProtoPartiallyEquals(get_after_finish_result[0],
                                      register_result[0],
                                      ignored_fields_with_state)
      self.assertProtoPartiallyEquals(get_after_finish_result[1],
                                      register_result[1],
                                      ignored_fields_with_state)
      self.assertEqual(metadata_store_pb2.Execution.State.COMPLETE,
                       get_after_finish_result[0].last_known_state)
      self.assertEqual(metadata_store_pb2.Execution.State.NEW,
                       get_after_finish_result[1].last_known_state)

      # Assert that we can retrieve the output artifact registered by
      # finish_external_execution.
      events = m.store.get_events_by_execution_ids([execution_id])
      self.assertLen(events, 1)
      self.assertEqual(metadata_store_pb2.Event.Type.OUTPUT, events[0].type)
      artifacts = m.store.get_artifacts_by_id([events[0].artifact_id])
      self.assertLen(artifacts, 1)
      self.assertEqual(uri, artifacts[0].uri)
      self.assertEqual(metadata_store_pb2.Artifact.State.LIVE,
                       artifacts[0].state)

  @mock.patch.object(outputs_utils, 'OutputsResolver')
  def test_external_execution_state_transitions(self, outputs_resolver):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline_with_example_gen('test_pipeline')
      node_uid = task_lib.NodeUid.from_pipeline_node(
          pipeline, pipeline.nodes[0].pipeline_node)
      pipeline_state = pipeline_ops.initiate_pipeline_start(m, pipeline)
      pipeline_run_id = pipeline_state.pipeline_run_id
      pipeline_ops.get_external_executions(m, node_uid, pipeline_run_id)
      register_result = pipeline_ops.register_external_executions(
          m, node_uid, pipeline_state.pipeline_run_id, [{}])
      execution_id = register_result[0].id
      outputs_resolver.return_value.get_executor_output_uri.return_value = ''
      outputs_resolver.return_value.get_stateful_working_directory.return_value = ''
      outputs_resolver.return_value.make_tmp_dir.return_value = ''
      # Cannot finish the execution before it starts.
      with self.assertRaises(
          status_lib.StatusNotOkError,
          msg='Expected execution state to be RUNNING'):
        pipeline_ops.finish_external_execution(
            m, node_uid, pipeline_run_id, execution_id,
            execution_result_pb2.ExecutorOutput())
      execution_invocation = pipeline_ops.start_external_execution(
          m, node_uid, pipeline_run_id, execution_id)
      # Can start an execution that is already started.
      second_execution_invocation = pipeline_ops.start_external_execution(
          m, node_uid, pipeline_run_id, execution_id)
      self.assertEqual(execution_invocation, second_execution_invocation)
      pipeline_ops.finish_external_execution(
          m, node_uid, pipeline_run_id, execution_id,
          execution_result_pb2.ExecutorOutput())
      mlmd_state.clear_in_memory_state()
      # Cannot start the execution after it finishes.
      with self.assertRaises(
          status_lib.StatusNotOkError,
          msg='Expected execution state to be NEW or RUNNING'):
        pipeline_ops.start_external_execution(m, node_uid, pipeline_run_id,
                                              execution_id)

  @mock.patch.object(outputs_utils, 'OutputsResolver')
  def test_finish_external_execution_with_error(self, outputs_resolver):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline_with_example_gen('test_pipeline')
      node_uid = task_lib.NodeUid.from_pipeline_node(
          pipeline, pipeline.nodes[0].pipeline_node)
      pipeline_state = pipeline_ops.initiate_pipeline_start(m, pipeline)
      pipeline_run_id = pipeline_state.pipeline_run_id
      register_result = pipeline_ops.register_external_executions(
          m, node_uid, pipeline_state.pipeline_run_id, [{}])
      execution_id = register_result[0].id
      outputs_resolver.return_value.get_executor_output_uri.return_value = ''
      outputs_resolver.return_value.get_stateful_working_directory.return_value = ''
      outputs_resolver.return_value.make_tmp_dir.return_value = ''
      pipeline_ops.start_external_execution(m, node_uid, pipeline_run_id,
                                            execution_id)
      # Calling finish_external_execution with an error status should register
      # a failed execution with that error status.
      pipeline_ops.finish_external_execution(
          m, node_uid, pipeline_run_id, execution_id,
          execution_result_pb2.ExecutorOutput(
              execution_result=execution_result_pb2.ExecutionResult(
                  code=status_lib.Code.FAILED_PRECONDITION,
                  result_message='test error message')))
      get_result = pipeline_ops.get_external_executions(m, node_uid,
                                                        pipeline_run_id)
      self.assertLen(get_result, 1)
      self.assertEqual(metadata_store_pb2.Execution.FAILED,
                       get_result[0].last_known_state)
      self.assertIn(
          'test error message',
          get_result[0].custom_properties['__execution_result__'].string_value)

  def test_external_executions_after_restarting_pipeline(self):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline_with_example_gen('test_pipeline')
      node_uid = task_lib.NodeUid.from_pipeline_node(
          pipeline, pipeline.nodes[0].pipeline_node)
      pipeline_state = pipeline_ops.initiate_pipeline_start(m, pipeline)
      pipeline_run_id = pipeline_state.pipeline_run_id
      pipeline_ops.get_external_executions(m, node_uid, pipeline_run_id)
      register_result = pipeline_ops.register_external_executions(
          m, node_uid, pipeline_state.pipeline_run_id, [{}])
      execution_id = register_result[0].id
      # Restart the pipeline.
      with pipeline_ops._PIPELINE_OPS_LOCK:
        with pipeline_state:
          pipeline_state.set_pipeline_execution_state(
              metadata_store_pb2.Execution.CANCELED)
      pipeline.runtime_spec.pipeline_run_id.field_value.string_value = 'run1'
      pipeline_ops.initiate_pipeline_start(m, pipeline)
      # Attempting to operate on an executions from the previous pipeline run
      # should result in an error.
      error_message = ('unable to find an active pipeline run for '
                       'pipeline_run_id: run0')
      with self.assertRaises(status_lib.StatusNotOkError, msg=error_message):
        pipeline_ops.register_external_executions(
            m, node_uid, pipeline_state.pipeline_run_id, [{}])
      with self.assertRaises(status_lib.StatusNotOkError, msg=error_message):
        pipeline_ops.get_external_executions(m, node_uid, pipeline_run_id)
      with self.assertRaises(status_lib.StatusNotOkError, msg=error_message):
        pipeline_ops.start_external_execution(m, node_uid, pipeline_run_id,
                                              execution_id)
      with self.assertRaises(status_lib.StatusNotOkError, msg=error_message):
        pipeline_ops.finish_external_execution(
            m, node_uid, pipeline_run_id, execution_id,
            execution_result_pb2.ExecutorOutput())


if __name__ == '__main__':
  tf.test.main()
