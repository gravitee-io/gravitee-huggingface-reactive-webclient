/*
 * Copyright © 2015 The Gravitee team (http://gravitee.io)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.gravitee.reactive.webclient.api;

import java.util.List;

/**
 * Metadata for a HuggingFace model repository, as returned by
 * {@code GET /api/models/{modelName}}.
 *
 * <p>Used for memory estimation pre-flight checks (via {@link SafetensorsInfo})
 * and for auto-discovery of model files (via {@link ModelFileInfo}).
 *
 * @param modelId        The full model identifier, e.g. {@code "meta-llama/Llama-3.1-8B"}.
 * @param gated          Whether the model requires accepting a licence gate.
 * @param isPrivate      Whether the model repository is private.
 * @param files          All files present in the repository with their sizes.
 * @param safetensors    Parameter count and dtype breakdown, or {@code null} if the model
 *                       does not publish safetensors metadata.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public record ModelInfo(String modelId, boolean gated, boolean isPrivate, List<ModelFileInfo> files, SafetensorsInfo safetensors) {
    /**
     * Returns {@code true} if safetensors parameter metadata is available.
     * When {@code false}, memory estimation will fall back to config.json parsing.
     */
    public boolean hasSafetensorsInfo() {
        return safetensors != null && safetensors.total() > 0;
    }
}
