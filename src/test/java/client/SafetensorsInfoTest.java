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
package client;

import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.reactive.webclient.api.SafetensorsInfo;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

class SafetensorsInfoTest {

    @Nested
    @DisplayName("estimateWeightBytes() with dominant dtype from parameters map")
    class DominantDtype {

        @Test
        @DisplayName("BF16 dominant (Qwen3-0.6B: 751_632_384 params): 2 bytes per param")
        void bf16_dominant_qwen3() {
            // Real safetensors metadata from Qwen/Qwen3-0.6B
            SafetensorsInfo info = new SafetensorsInfo(Map.of("BF16", 751_632_384L), 751_632_384L);
            assertThat(info.estimateWeightBytes("auto")).isEqualTo(751_632_384L * 2);
        }

        @Test
        @DisplayName("F32 dominant: 4 bytes per param")
        void f32_dominant() {
            SafetensorsInfo info = new SafetensorsInfo(Map.of("F32", 1_000_000L), 1_000_000L);
            assertThat(info.estimateWeightBytes("auto")).isEqualTo(1_000_000L * 4);
        }

        @Test
        @DisplayName("F16 dominant: 2 bytes per param")
        void f16_dominant() {
            SafetensorsInfo info = new SafetensorsInfo(Map.of("F16", 7_000_000_000L), 7_000_000_000L);
            assertThat(info.estimateWeightBytes("auto")).isEqualTo(7_000_000_000L * 2);
        }

        @Test
        @DisplayName("I8 dominant: 1 byte per param")
        void int8_dominant() {
            SafetensorsInfo info = new SafetensorsInfo(Map.of("I8", 3_000_000_000L), 3_000_000_000L);
            assertThat(info.estimateWeightBytes("auto")).isEqualTo(3_000_000_000L * 1);
        }

        @Test
        @DisplayName("mixed dtypes: uses dominant (highest count)")
        void mixed_dtypes_uses_dominant() {
            // BF16 dominant (7B), small F32 layer norms (30M)
            SafetensorsInfo info = new SafetensorsInfo(Map.of("BF16", 7_000_000_000L, "F32", 30_000_000L), 7_030_000_000L);
            // Dominant is BF16 -> 2 bytes per param
            assertThat(info.estimateWeightBytes("auto")).isEqualTo(7_030_000_000L * 2);
        }
    }

    @Nested
    @DisplayName("estimateWeightBytes() with runtime dtype fallback")
    class RuntimeDtypeFallback {

        @Test
        @DisplayName("null parameters map: falls back to runtime dtype")
        void null_parameters_map() {
            SafetensorsInfo info = new SafetensorsInfo(null, 8_000_000_000L);
            // "auto" -> 2 bytes default
            assertThat(info.estimateWeightBytes("auto")).isEqualTo(8_000_000_000L * 2);
        }

        @Test
        @DisplayName("empty parameters map: falls back to runtime dtype")
        void empty_parameters_map() {
            SafetensorsInfo info = new SafetensorsInfo(Map.of(), 8_000_000_000L);
            assertThat(info.estimateWeightBytes("auto")).isEqualTo(8_000_000_000L * 2);
        }

        @ParameterizedTest
        @CsvSource({ "auto,2", "float32,4", "fp32,4", "float16,2", "fp16,2", "bfloat16,2", "bf16,2" })
        @DisplayName("dtypeBytes maps runtime dtype strings correctly")
        void runtime_dtype_mapping(String dtype, int expectedBytesPerParam) {
            SafetensorsInfo info = new SafetensorsInfo(null, 1_000_000L);
            assertThat(info.estimateWeightBytes(dtype)).isEqualTo(1_000_000L * expectedBytesPerParam);
        }

        @Test
        @DisplayName("null runtime dtype defaults to 2 bytes (auto)")
        void null_runtime_dtype() {
            SafetensorsInfo info = new SafetensorsInfo(null, 5_000_000L);
            assertThat(info.estimateWeightBytes(null)).isEqualTo(5_000_000L * 2);
        }

        @Test
        @DisplayName("unknown runtime dtype defaults to 2 bytes")
        void unknown_runtime_dtype() {
            SafetensorsInfo info = new SafetensorsInfo(null, 5_000_000L);
            assertThat(info.estimateWeightBytes("some_unknown_dtype")).isEqualTo(5_000_000L * 2);
        }
    }

    @Nested
    @DisplayName("dtypeBytesFromHfKey mapping")
    class HfKeyMapping {

        @Test
        @DisplayName("F8/FLOAT8 family: 1 byte per param")
        void f8_family() {
            SafetensorsInfo info = new SafetensorsInfo(Map.of("F8", 1_000L), 1_000L);
            assertThat(info.estimateWeightBytes("auto")).isEqualTo(1_000L);

            info = new SafetensorsInfo(Map.of("E4M3", 1_000L), 1_000L);
            assertThat(info.estimateWeightBytes("auto")).isEqualTo(1_000L);

            info = new SafetensorsInfo(Map.of("E5M2", 1_000L), 1_000L);
            assertThat(info.estimateWeightBytes("auto")).isEqualTo(1_000L);
        }

        @Test
        @DisplayName("I4/INT4: 0 bytes per param (handled as 0.5 in code)")
        void int4_is_zero() {
            // I4 returns 0 from dtypeBytesFromHfKey -- total * 0 = 0
            SafetensorsInfo info = new SafetensorsInfo(Map.of("I4", 1_000_000L), 1_000_000L);
            assertThat(info.estimateWeightBytes("auto")).isEqualTo(0);
        }

        @Test
        @DisplayName("unknown HF key: falls back to runtime dtype bytes")
        void unknown_hf_key_fallback() {
            SafetensorsInfo info = new SafetensorsInfo(Map.of("WEIRD_TYPE", 1_000_000L), 1_000_000L);
            // Fallback = dtypeBytes("auto") = 2
            assertThat(info.estimateWeightBytes("auto")).isEqualTo(1_000_000L * 2);
        }
    }

    @Nested
    @DisplayName("Record accessors")
    class RecordAccessors {

        @Test
        @DisplayName("parameters() and total() return correct values")
        void accessors() {
            Map<String, Long> params = Map.of("BF16", 7_241_748_480L);
            SafetensorsInfo info = new SafetensorsInfo(params, 7_241_748_480L);
            assertThat(info.parameters()).isEqualTo(params);
            assertThat(info.total()).isEqualTo(7_241_748_480L);
        }
    }
}
