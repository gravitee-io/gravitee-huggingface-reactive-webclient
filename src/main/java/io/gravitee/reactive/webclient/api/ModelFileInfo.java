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

/**
 * Metadata for a single file in a HuggingFace model repository,
 * derived from the {@code siblings[]} array in the model info API response.
 *
 * @param name      The relative file path within the repository, e.g. {@code "config.json"}
 *                  or {@code "onnx/model.onnx"}.
 * @param sizeBytes The file size in bytes as reported by the HuggingFace API,
 *                  or {@code -1} if not available.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public record ModelFileInfo(String name, long sizeBytes) {}
