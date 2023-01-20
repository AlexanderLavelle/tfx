# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Tests for tfx.dsl.input_resolution.ops.latest_policy_model_op."""

from absl.testing import parameterized

import tensorflow as tf

from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.input_resolution.ops import test_utils
from tfx.types import artifact_utils
from tfx.utils import test_case_utils as mlmd_mixins


class LatestPolicyModelOpTest(
    tf.test.TestCase, parameterized.TestCase, mlmd_mixins.MlmdMixins
):

  def _latest_policy_model(self, child_artifact_type: str = ''):
    """Run the LatestPolicyModel ResolverOp."""
    return test_utils.run_resolver_op(
        ops.LatestPolicyModel,
        self.artifacts,
        context=resolver_op.Context(store=self.store),
        child_artifact_type=child_artifact_type,
    )

  def _prepare_tfx_artifact(self, artifact_type: str):
    """Adds a single artifact to MLMD and returns the TFleX Artifact object."""
    artifact = self.put_artifact(artifact_type)
    artifact_type = self.store.get_artifact_type(artifact_type)
    return artifact_utils.deserialize_artifact(artifact_type, artifact)

  def _unwrap_tfx_artifact(self, artifact: types.Artifact):
    """Return the underlying MLMD Artifact of a TFleX Artifact object."""
    return [artifact.mlmd_artifact]

  def _evaluator_bless_model(self, model: types.Artifact):
    """Add an Execution to MLMD where the Evaluator blesses the model."""
    model_blessing = self._prepare_tfx_artifact('ModelBlessing')
    self.put_execution(
        'Evaluator',
        inputs={'model': self._unwrap_tfx_artifact(model)},
        outputs={'blessing': self._unwrap_tfx_artifact(model_blessing)},
    )

  def _infra_validator_bless_model(self, model: types.Artifact):
    """Add an Execution to MLMD where the InfraValidator blesses the model."""
    model_infra_blessing = self._prepare_tfx_artifact('ModelInfraBlessing')
    self.put_execution(
        'InfraValidator',
        inputs={'model': self._unwrap_tfx_artifact(model)},
        outputs={'result': self._unwrap_tfx_artifact(model_infra_blessing)},
    )

  def _push_model(self, model: types.Artifact):
    """Add an Execution to MLMD where the Pusher pushes the model."""
    model_push = self._prepare_tfx_artifact('ModelPush')
    self.put_execution(
        'Pusher',
        inputs={'model': self._unwrap_tfx_artifact(model)},
        outputs={'model_push': self._unwrap_tfx_artifact(model_push)},
    )

  def setUp(self):
    super().setUp()
    self.init_mlmd()

    self.model_1 = self._prepare_tfx_artifact('Model')
    self.model_2 = self._prepare_tfx_artifact('Model')
    self.model_3 = self._prepare_tfx_artifact('Model')

    self.artifacts = [self.model_1, self.model_2, self.model_3]

  def testLatestPolicyModelOpTest_EmptyInput(self):
    actual = test_utils.run_resolver_op(
        ops.LatestPolicyModel, [], context=resolver_op.Context(store=self.store)
    )
    self.assertEqual(actual, [])

  def testLatestPolicyModelOpTest_NonModelInput(self):
    model_blessing = self._prepare_tfx_artifact('ModelBlessing')
    actual = test_utils.run_resolver_op(
        ops.LatestPolicyModel,
        [model_blessing],
        context=resolver_op.Context(store=self.store),
    )
    self.assertEqual(actual, [])

  def testLatestPolicyModelOpTest_LatestTrainedModel(self):
    actual = self._latest_policy_model()
    self.assertEqual(actual, [self.model_3])

  def testLatestPolicyModelOp_SeqeuntialExecutions_LatestModelChanges(self):
    actual = self._latest_policy_model('ModelBlessing')
    self.assertEqual(actual, [])

    # Insert spurious Executions.
    self._push_model(self.model_1)
    self._infra_validator_bless_model(self.model_2)
    self._push_model(self.model_3)

    self._evaluator_bless_model(self.model_1)
    actual = self._latest_policy_model('ModelBlessing')
    self.assertEqual(actual, [self.model_1])

    self._evaluator_bless_model(self.model_3)
    actual = self._latest_policy_model('ModelBlessing')
    self.assertEqual(actual, [self.model_3])

    # model_3 should still be the latest model, since it is the latest created.
    self._evaluator_bless_model(self.model_2)
    actual = self._latest_policy_model('ModelBlessing')
    self.assertEqual(actual, [self.model_3])

  @parameterized.parameters(
      (['m1'], [], [], 'ModelBlessing', ['m1']),
      (['m1'], [], [], 'ModelInfraBlessing', []),
      (['m1'], [], [], 'ModelPush', []),
      (['m1', 'm2', 'm3'], ['m2', 'm3'], ['m3'], 'ModelBlessing', ['m3']),
      (['m1', 'm2', 'm3'], ['m2', 'm3'], ['m3'], 'ModelInfraBlessing', ['m3']),
      (['m1', 'm2', 'm3'], ['m2', 'm3'], ['m3'], 'ModelPush', ['m3']),
      (['m1', 'm2', 'm3'], ['m2', 'm3'], ['m1'], 'ModelPush', ['m1']),
      (['m2', 'm1'], [], [], 'ModelBlessing', ['m2']),
  )
  def testLatestPolicyModelOp(
      self,
      eval_models: list[types.Artifact],
      infra_val_models: list[types.Artifact],
      push_models: list[types.Artifact],
      child_artifact_type: str,
      expected: list[types.Artifact],
  ):
    str_to_model = {
        'm1': self.model_1,
        'm2': self.model_2,
        'm3': self.model_3,
    }

    for model in eval_models:
      self._evaluator_bless_model(str_to_model[model])

    for model in infra_val_models:
      self._infra_validator_bless_model(str_to_model[model])

    for model in push_models:
      self._push_model(str_to_model[model])

    actual = self._latest_policy_model(child_artifact_type)
    self.assertEqual(actual, [str_to_model[e] for e in expected])


if __name__ == '__main__':
  tf.test.main()
