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

import java.util.Map;

/**
 * Safetensors parameter metadata for a HuggingFace model, as published
 * in the {@code safetensors} field of the model info API response.
 *
 * <p>Provides the exact parameter count per dtype, which is used for
 * accurate VRAM weight-size estimation without requiring the model weights
 * to be downloaded.
 *
 * <p>Example: {@code parameters={"BF16": 7241748480}, total=7241748480}
 *
 * @param parameters A map from dtype name (e.g. {@code "BF16"}, {@code "F32"}) to
 *                   the number of parameters stored in that dtype.
 * @param total      The total number of parameters across all dtypes.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public record SafetensorsInfo(Map<String, Long> parameters, long total) {
    /**
     * Estimates the weight size in bytes for a given runtime dtype.
     *
     * <p>Uses the stored dtype breakdown when available, otherwise falls back
     * to {@code total × dtypeBytes}.
     *
     * @param runtimeDtype the dtype string as configured, e.g. {@code "auto"},
     *                     {@code "bfloat16"}, {@code "float16"}, {@code "float32"}.
     * @return estimated weight bytes
     */
    public long estimateWeightBytes(String runtimeDtype) {
        int bytesPerParam = dtypeBytes(runtimeDtype);

        // If parameters map has a single dominant dtype, use its exact count
        if (parameters != null && !parameters.isEmpty()) {
            // Pick the most common dtype by count
            long dominant = parameters.values().stream().mapToLong(Long::longValue).max().orElse(total);
            // Determine its byte width from the map key
            String dominantDtype = parameters
                .entrySet()
                .stream()
                .filter(e -> e.getValue() == dominant)
                .map(Map.Entry::getKey)
                .findFirst()
                .orElse(runtimeDtype);
            bytesPerParam = dtypeBytesFromHfKey(dominantDtype, bytesPerParam);
        }

        return total * bytesPerParam;
    }

    // Maps HuggingFace dtype key names to byte widths
    private static int dtypeBytesFromHfKey(String hfKey, int fallback) {
        if (hfKey == null) return fallback;
        return switch (hfKey.toUpperCase()) {
            case "F32", "FLOAT32" -> 4;
            case "F16", "FLOAT16", "BF16", "BFLOAT16" -> 2;
            case "F8", "FLOAT8", "E4M3", "E5M2" -> 1;
            case "I8", "INT8" -> 1;
            case "I4", "INT4" -> 0; // handled below as 0.5
            default -> fallback;
        };
    }

    // Maps runtime dtype config strings to byte widths
    private static int dtypeBytes(String dtype) {
        if (dtype == null || dtype.equalsIgnoreCase("auto")) return 2; // assume bf16
        return switch (dtype.toLowerCase()) {
            case "float32", "fp32" -> 4;
            case "float16", "fp16", "bfloat16", "bf16" -> 2;
            default -> 2;
        };
    }
}
