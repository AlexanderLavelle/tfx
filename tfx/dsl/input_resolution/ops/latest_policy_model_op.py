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
"""Module for LatestPolicyModel operator."""

import collections

from typing import Sequence

from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.types import standard_artifacts

from ml_metadata.proto import metadata_store_pb2


_VALID_CHILD_ARTIFACT_TYPES = [
    'ModelBlessing',
    'ModelInfraBlessing',
    'ModelPush',
]


class LatestPolicyModel(
    resolver_op.ResolverOp,
    canonical_name='tfx.LatestPolicyModel',
    arg_data_types=(resolver_op.DataType.ARTIFACT_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST,
):
  """LatestPolicyModel operator."""

  # The child artifact type. If not set, the latest trained model will be
  # returned. See _VALID_CHILD_ARTIFACT_TYPES for other options.
  child_artifact_type = resolver_op.Property(type=str, default='')

  def apply(
      self, input_list: Sequence[types.Artifact]
  ) -> Sequence[types.Artifact]:
    """Finds the latest created model via a certain policy."""
    # Only consider Model artifacts.
    models = [
        artifact
        for artifact in input_list
        if isinstance(artifact, standard_artifacts.Model)
    ]
    if not models:
      return []

    # Sort the models from from latest created to oldest.
    models.sort(  # pytype: disable=attribute-error
        key=lambda a: (a.mlmd_artifact.create_time_since_epoch, a.id)
    )
    models.reverse()

    # Return the latest trained model if child_artifact_type is not set.
    if not self.child_artifact_type:
      return [models[0]]

    # LatestPolicyModel only supports getting the latest model blessed by the
    # Evaluator or InfraValdator, or the latest pushed model.
    if self.child_artifact_type not in _VALID_CHILD_ARTIFACT_TYPES:
      raise ValueError(
          f'child_artifact_type must be one of {_VALID_CHILD_ARTIFACT_TYPES}, '
          f'but was set to {self.child_artifact_type}.'
      )

    # In MLMD, two artifacts are related by:
    #
    #       Event 1           Event 2
    # Model ------> Execution ------> Artifact B
    #
    # Artifact B can be:
    # 1. ModelBlessing output artifact from an Evaluator.
    # 2. ModelInfraBlessing output artifact from an InfraValidator.
    # 3. ModelPush output artifact from a Pusher.
    #
    # We query MLMD to get a list of candidate model artifact ids that have
    # a child artifact of type child_artifact_type. Note we perform batch
    # queries to reduce the number round trips to the database.

    # Get all Executions in MLMD associated with the Model artifacts.
    artifact_ids = [m.id for m in models]
    executions_ids = set()
    executions_id_to_artifact_id = {}
    for event in self.context.store.get_events_by_artifact_ids(artifact_ids):
      if event.type == metadata_store_pb2.Event.INPUT:
        executions_id_to_artifact_id[event.execution_id] = event.artifact_id
        executions_ids.add(event.execution_id)

    # Get all artifact ids associated with an OUTPUT Event in each Execution.
    # These ids correspond to descendant artifacts 1 hop distance away from the
    # Model.
    child_artifact_ids = set()
    child_artifact_id_to_model_artifact_id = {}
    for event in self.context.store.get_events_by_execution_ids(executions_ids):
      if event.type == metadata_store_pb2.Event.OUTPUT:
        child_artifact_ids.add(event.artifact_id)
        child_artifact_id_to_model_artifact_id[event.artifact_id] = (
            executions_id_to_artifact_id[event.execution_id]
        )

    # Get the type_id of the child artifacts.
    child_type_ids = set()
    child_type_id_to_artifact_ids = collections.defaultdict(set)
    for artifact in self.context.store.get_artifacts_by_id(child_artifact_ids):
      child_type_id_to_artifact_ids[artifact.type_id].add(artifact.id)
      child_type_ids.add(artifact.type_id)

    # Get a list of Model artifact ids that have a child artifact with the
    # matching child_artifact_type.
    candidate_model_artifact_ids = set()
    for artifact_type in self.context.store.get_artifact_types_by_id(
        child_type_ids
    ):
      if artifact_type.name == self.child_artifact_type:
        for child_artifact_id in child_type_id_to_artifact_ids[
            artifact_type.id
        ]:
          candidate_model_artifact_ids.add(
              child_artifact_id_to_model_artifact_id[child_artifact_id]
          )
        break

    # Return the latest model that has a child artifact we are looking for.
    # Recall that models is already sorted from latest to oldest created.
    for model in models:
      if model.id in candidate_model_artifact_ids:
        return [model]

    # Return nothing if no Model artifact has the corresponding child artifact
    # 1 hop distance away.
    return []
