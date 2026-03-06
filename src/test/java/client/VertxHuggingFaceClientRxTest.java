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

import static com.github.tomakehurst.wiremock.client.WireMock.aResponse;
import static com.github.tomakehurst.wiremock.client.WireMock.equalTo;
import static com.github.tomakehurst.wiremock.client.WireMock.get;
import static com.github.tomakehurst.wiremock.client.WireMock.okJson;
import static com.github.tomakehurst.wiremock.client.WireMock.stubFor;
import static com.github.tomakehurst.wiremock.client.WireMock.urlEqualTo;
import static com.github.tomakehurst.wiremock.client.WireMock.urlPathEqualTo;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.ThrowableAssert.catchThrowable;

import com.github.tomakehurst.wiremock.client.WireMock;
import com.github.tomakehurst.wiremock.junit5.WireMockTest;
import io.gravitee.reactive.webclient.api.ModelInfo;
import io.gravitee.reactive.webclient.huggingface.client.VertxHuggingFaceClientRx;
import io.vertx.core.file.OpenOptions;
import io.vertx.ext.web.client.WebClientOptions;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.file.AsyncFile;
import io.vertx.rxjava3.ext.web.client.WebClient;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@WireMockTest(httpPort = 8080)
class VertxHuggingFaceClientRxTest {

    VertxHuggingFaceClientRx huggingFaceClient;
    VertxHuggingFaceClientRx huggingFaceClientWithToken;

    @BeforeEach
    void setUp() {
        var webClient = WebClient.create(
            Vertx.vertx(),
            new WebClientOptions().setDefaultHost("localhost").setDefaultPort(8080).setSsl(false).setLogActivity(true)
        );
        this.huggingFaceClient = new VertxHuggingFaceClientRx(webClient);
        this.huggingFaceClientWithToken = new VertxHuggingFaceClientRx(webClient, "hf-test-token-123");
    }

    // -------------------------------------------------------------------------
    // listModelFiles (delegates to fetchModelInfo internally)
    // -------------------------------------------------------------------------

    @Test
    @DisplayName("listModelFiles: returns file names from siblings array")
    void shouldReturnListOfModelFiles() {
        stubFor(
            get(urlEqualTo("/api/models/minuva/MiniLMv2-toxic-jigsaw-onnx"))
                .willReturn(
                    okJson(
                        """
                        {
                          "siblings": [
                            { "rfilename": "config.json", "size": 123 },
                            { "rfilename": "tokenizer.json", "size": 456 },
                            { "rfilename": "model_optimized_quantized.onnx", "size": 789 }
                          ]
                        }
                        """
                    )
                )
        );

        List<String> files = huggingFaceClient.listModelFiles("minuva/MiniLMv2-toxic-jigsaw-onnx").toList().blockingGet();

        assertThat(files).containsExactlyInAnyOrder("config.json", "tokenizer.json", "model_optimized_quantized.onnx");
    }

    @Test
    @DisplayName("listModelFiles: completes empty on API error (onErrorComplete)")
    void shouldCompleteWithoutErrorWhenApiReturns404() {
        stubFor(get(urlEqualTo("/api/models/missing-model")).willReturn(aResponse().withStatus(404).withBody("{}")));

        List<String> result = huggingFaceClient.listModelFiles("missing-model").toList().blockingGet();

        assertThat(result).isEmpty();
    }

    // -------------------------------------------------------------------------
    // fetchModelInfo
    // -------------------------------------------------------------------------

    @Test
    @DisplayName("fetchModelInfo: parses gated flag, safetensors, and file sizes")
    void shouldFetchFullModelInfo() {
        stubFor(
            get(urlEqualTo("/api/models/meta-llama/Llama-3.1-8B"))
                .willReturn(
                    okJson(
                        """
                        {
                          "gated": true,
                          "private": false,
                          "siblings": [
                            { "rfilename": "config.json", "size": 1024 },
                            { "rfilename": "model.safetensors", "size": 14483455928 }
                          ],
                          "safetensors": {
                            "parameters": { "BF16": 8030261248 },
                            "total": 8030261248
                          }
                        }
                        """
                    )
                )
        );

        ModelInfo info = huggingFaceClient.fetchModelInfo("meta-llama/Llama-3.1-8B").blockingGet();

        assertThat(info.modelId()).isEqualTo("meta-llama/Llama-3.1-8B");
        assertThat(info.gated()).isTrue();
        assertThat(info.isPrivate()).isFalse();
        assertThat(info.files()).hasSize(2);
        assertThat(info.files().get(0).name()).isEqualTo("config.json");
        assertThat(info.files().get(0).sizeBytes()).isEqualTo(1024L);
        assertThat(info.hasSafetensorsInfo()).isTrue();
        assertThat(info.safetensors().total()).isEqualTo(8030261248L);
        assertThat(info.safetensors().parameters()).containsEntry("BF16", 8030261248L);
    }

    @Test
    @DisplayName("fetchModelInfo: safetensors is null when field absent")
    void shouldParseModelInfoWithNoSafetensors() {
        stubFor(
            get(urlEqualTo("/api/models/some-org/onnx-model"))
                .willReturn(
                    okJson(
                        """
                        {
                          "gated": false,
                          "private": false,
                          "siblings": [
                            { "rfilename": "model.onnx", "size": 500000 }
                          ]
                        }
                        """
                    )
                )
        );

        ModelInfo info = huggingFaceClient.fetchModelInfo("some-org/onnx-model").blockingGet();

        assertThat(info.hasSafetensorsInfo()).isFalse();
        assertThat(info.safetensors()).isNull();
        assertThat(info.files()).hasSize(1);
    }

    // -------------------------------------------------------------------------
    // Auth header
    // -------------------------------------------------------------------------

    @Test
    @DisplayName("fetchModelInfo: sends Authorization header when token is set")
    void shouldSendAuthorizationHeaderWhenTokenSet() {
        stubFor(
            get(urlEqualTo("/api/models/meta-llama/Llama-3.1-8B"))
                .withHeader("Authorization", equalTo("Bearer hf-test-token-123"))
                .willReturn(
                    okJson(
                        """
                        {
                          "gated": true,
                          "private": false,
                          "siblings": [],
                          "safetensors": { "parameters": {}, "total": 0 }
                        }
                        """
                    )
                )
        );

        ModelInfo info = huggingFaceClientWithToken.fetchModelInfo("meta-llama/Llama-3.1-8B").blockingGet();

        assertThat(info).isNotNull();
        assertThat(info.gated()).isTrue();
        // Verify WireMock matched (would have returned 404 otherwise and thrown)
        WireMock.verify(
            1,
            WireMock
                .getRequestedFor(urlEqualTo("/api/models/meta-llama/Llama-3.1-8B"))
                .withHeader("Authorization", equalTo("Bearer hf-test-token-123"))
        );
    }

    @Test
    @DisplayName("fetchModelInfo: no Authorization header when no token")
    void shouldNotSendAuthorizationHeaderWhenNoToken() {
        stubFor(
            get(urlEqualTo("/api/models/public-org/public-model"))
                .willReturn(okJson("""
                    { "gated": false, "private": false, "siblings": [] }
                    """))
        );

        huggingFaceClient.fetchModelInfo("public-org/public-model").blockingGet();

        WireMock.verify(1, WireMock.getRequestedFor(urlEqualTo("/api/models/public-org/public-model")).withoutHeader("Authorization"));
    }

    @Test
    @DisplayName("downloadModelFile: sends Authorization header when token is set")
    void shouldSendAuthHeaderOnDownload() throws IOException {
        stubFor(
            get(urlPathEqualTo("/meta-llama/Llama-3.1-8B/resolve/main/config.json"))
                .withHeader("Authorization", equalTo("Bearer hf-test-token-123"))
                .withQueryParam("download", equalTo("true"))
                .willReturn(aResponse().withStatus(200).withBody("{}"))
        );

        Vertx vertx = Vertx.vertx();
        Path tempFile = Files.createTempFile("hf-auth-test-", ".json");
        AsyncFile asyncFile = vertx
            .fileSystem()
            .rxOpen(tempFile.toString(), new OpenOptions().setCreate(true).setWrite(true).setTruncateExisting(true))
            .blockingGet();

        try {
            huggingFaceClientWithToken.downloadModelFile("meta-llama/Llama-3.1-8B", "config.json", asyncFile).blockingAwait();
            WireMock.verify(
                1,
                WireMock
                    .getRequestedFor(urlPathEqualTo("/meta-llama/Llama-3.1-8B/resolve/main/config.json"))
                    .withHeader("Authorization", equalTo("Bearer hf-test-token-123"))
            );
        } finally {
            Files.deleteIfExists(tempFile);
        }
    }

    // -------------------------------------------------------------------------
    // downloadModelFile (existing coverage preserved)
    // -------------------------------------------------------------------------

    @Test
    @DisplayName("downloadModelFile: downloads file content successfully")
    void shouldDownloadModelFileSuccessfully() throws IOException {
        String modelName = "minuva/MiniLMv2-toxic-jigsaw-onnx";
        String fileName = "config.json";
        String fileContent = "{ \"model_type\": \"bert\" }";

        stubFor(
            get(urlPathEqualTo("/minuva/MiniLMv2-toxic-jigsaw-onnx/resolve/main/config.json"))
                .withQueryParam("download", equalTo("true"))
                .willReturn(aResponse().withStatus(200).withBody(fileContent))
        );

        Vertx vertx = Vertx.vertx();
        Path tempFile = Files.createTempFile("hf-test-", ".json");
        AsyncFile asyncFile = vertx
            .fileSystem()
            .rxOpen(tempFile.toString(), new OpenOptions().setCreate(true).setWrite(true).setTruncateExisting(true))
            .blockingGet();

        try {
            huggingFaceClient.downloadModelFile(modelName, fileName, asyncFile).blockingAwait();
            String result = Files.readString(tempFile);
            assertThat(result).isEqualTo(fileContent);
        } finally {
            Files.deleteIfExists(tempFile);
        }
    }

    // -------------------------------------------------------------------------
    // fetchFileAsJson
    // -------------------------------------------------------------------------

    /** Real Qwen3-0.6B config.json from HuggingFace Hub. */
    static final String QWEN3_CONFIG_JSON =
        """
        {
          "architectures": ["Qwen3ForCausalLM"],
          "attention_bias": false,
          "attention_dropout": 0.0,
          "bos_token_id": 151643,
          "eos_token_id": 151645,
          "head_dim": 128,
          "hidden_act": "silu",
          "hidden_size": 1024,
          "initializer_range": 0.02,
          "intermediate_size": 3072,
          "max_position_embeddings": 40960,
          "max_window_layers": 28,
          "model_type": "qwen3",
          "num_attention_heads": 16,
          "num_hidden_layers": 28,
          "num_key_value_heads": 8,
          "rms_norm_eps": 1e-06,
          "rope_scaling": null,
          "rope_theta": 1000000,
          "sliding_window": null,
          "tie_word_embeddings": true,
          "torch_dtype": "bfloat16",
          "transformers_version": "4.51.0",
          "use_cache": true,
          "use_sliding_window": false,
          "vocab_size": 151936
        }
        """;

    /** Real Qwen2.5-VL-3B config.json (vision model) from HuggingFace Hub — trimmed to key fields. */
    static final String QWEN25_VL_CONFIG_JSON =
        """
        {
          "architectures": ["Qwen2_5_VLForConditionalGeneration"],
          "hidden_size": 2048,
          "model_type": "qwen2_5_vl",
          "num_attention_heads": 16,
          "num_hidden_layers": 36,
          "num_key_value_heads": 2,
          "torch_dtype": "bfloat16",
          "vocab_size": 151936,
          "vision_config": {
            "depth": 32,
            "hidden_act": "silu",
            "hidden_size": 1280,
            "intermediate_size": 3420,
            "num_heads": 16,
            "in_chans": 3,
            "out_hidden_size": 2048,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "spatial_patch_size": 14,
            "window_size": 112,
            "tokens_per_second": 2,
            "temporal_patch_size": 2
          }
        }
        """;

    /** Real Qwen2-Audio-7B config.json (audio model) from HuggingFace Hub — trimmed to key fields. */
    static final String QWEN2_AUDIO_CONFIG_JSON =
        """
        {
          "architectures": ["Qwen2AudioForConditionalGeneration"],
          "model_type": "qwen2_audio",
          "audio_config": {
            "model_type": "qwen2_audio_encoder",
            "num_mel_bins": 128,
            "encoder_layers": 32,
            "encoder_attention_heads": 20,
            "encoder_ffn_dim": 5120,
            "d_model": 1280,
            "activation_function": "gelu",
            "max_source_positions": 1500
          },
          "text_config": {
            "intermediate_size": 11008,
            "max_position_embeddings": 8192,
            "model_type": "qwen2",
            "torch_dtype": "bfloat16",
            "vocab_size": 156032
          },
          "vocab_size": 156032
        }
        """;

    @Test
    @DisplayName("fetchFileAsJson: parses Qwen3-0.6B config.json with all architecture fields")
    void shouldFetchQwen3ConfigJson() {
        stubFor(get(urlPathEqualTo("/Qwen/Qwen3-0.6B/resolve/main/config.json")).willReturn(okJson(QWEN3_CONFIG_JSON)));

        var result = huggingFaceClient.fetchFileAsJson("Qwen/Qwen3-0.6B", "config.json").blockingGet();

        assertThat(result.getString("model_type")).isEqualTo("qwen3");
        assertThat(result.getInteger("hidden_size")).isEqualTo(1024);
        assertThat(result.getInteger("num_hidden_layers")).isEqualTo(28);
        assertThat(result.getInteger("num_key_value_heads")).isEqualTo(8);
        assertThat(result.getInteger("num_attention_heads")).isEqualTo(16);
        assertThat(result.getInteger("head_dim")).isEqualTo(128);
        assertThat(result.getString("torch_dtype")).isEqualTo("bfloat16");
        assertThat(result.getJsonArray("architectures").getString(0)).isEqualTo("Qwen3ForCausalLM");
        // Text-only model: no vision_config, no audio_config
        assertThat(result.containsKey("vision_config")).isFalse();
        assertThat(result.containsKey("audio_config")).isFalse();
    }

    @Test
    @DisplayName("fetchFileAsJson: detects Qwen2.5-VL vision_config (multimodal)")
    void shouldDetectVisionConfigFromQwen25VL() {
        stubFor(get(urlPathEqualTo("/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/config.json")).willReturn(okJson(QWEN25_VL_CONFIG_JSON)));

        var result = huggingFaceClient.fetchFileAsJson("Qwen/Qwen2.5-VL-3B-Instruct", "config.json").blockingGet();

        assertThat(result.containsKey("vision_config")).isTrue();
        assertThat(result.getJsonObject("vision_config").getInteger("hidden_size")).isEqualTo(1280);
        assertThat(result.getJsonObject("vision_config").getInteger("depth")).isEqualTo(32);
        assertThat(result.getString("model_type")).isEqualTo("qwen2_5_vl");
        assertThat(result.getInteger("num_hidden_layers")).isEqualTo(36);
        assertThat(result.containsKey("audio_config")).isFalse();
    }

    @Test
    @DisplayName("fetchFileAsJson: detects Qwen2-Audio audio_config (multimodal)")
    void shouldDetectAudioConfigFromQwen2Audio() {
        stubFor(get(urlPathEqualTo("/Qwen/Qwen2-Audio-7B-Instruct/resolve/main/config.json")).willReturn(okJson(QWEN2_AUDIO_CONFIG_JSON)));

        var result = huggingFaceClient.fetchFileAsJson("Qwen/Qwen2-Audio-7B-Instruct", "config.json").blockingGet();

        assertThat(result.containsKey("audio_config")).isTrue();
        assertThat(result.getJsonObject("audio_config").getInteger("encoder_layers")).isEqualTo(32);
        assertThat(result.getJsonObject("audio_config").getInteger("d_model")).isEqualTo(1280);
        assertThat(result.getString("model_type")).isEqualTo("qwen2_audio");
        assertThat(result.containsKey("vision_config")).isFalse();
    }

    @Test
    @DisplayName("fetchFileAsJson: throws on HTTP 404")
    void shouldThrowOnFetchFileAsJson404() {
        stubFor(
            get(urlPathEqualTo("/missing-org/missing-model/resolve/main/config.json"))
                .willReturn(aResponse().withStatus(404).withBody("Not found"))
        );

        Throwable thrown = catchThrowable(() -> huggingFaceClient.fetchFileAsJson("missing-org/missing-model", "config.json").blockingGet()
        );

        assertThat(thrown).isInstanceOf(IllegalStateException.class);
        assertThat(thrown.getMessage()).contains("404");
        assertThat(thrown.getMessage()).contains("config.json");
    }

    @Test
    @DisplayName("fetchFileAsJson: throws on HTTP 500")
    void shouldThrowOnFetchFileAsJson500() {
        stubFor(
            get(urlPathEqualTo("/org/model/resolve/main/config.json")).willReturn(aResponse().withStatus(500).withBody("Internal error"))
        );

        Throwable thrown = catchThrowable(() -> huggingFaceClient.fetchFileAsJson("org/model", "config.json").blockingGet());

        assertThat(thrown).isInstanceOf(RuntimeException.class);
    }

    @Test
    @DisplayName("fetchFileAsJson: sends Authorization header when token is set")
    void shouldSendAuthHeaderOnFetchFileAsJson() {
        stubFor(
            get(urlPathEqualTo("/Qwen/Qwen3-0.6B/resolve/main/config.json"))
                .withHeader("Authorization", equalTo("Bearer hf-test-token-123"))
                .willReturn(okJson(QWEN3_CONFIG_JSON))
        );

        var result = huggingFaceClientWithToken.fetchFileAsJson("Qwen/Qwen3-0.6B", "config.json").blockingGet();

        assertThat(result.getInteger("num_hidden_layers")).isEqualTo(28);
        WireMock.verify(
            1,
            WireMock
                .getRequestedFor(urlPathEqualTo("/Qwen/Qwen3-0.6B/resolve/main/config.json"))
                .withHeader("Authorization", equalTo("Bearer hf-test-token-123"))
        );
    }

    @Test
    @DisplayName("fetchFileAsJson: no Authorization header when no token")
    void shouldNotSendAuthHeaderOnFetchFileAsJsonWithoutToken() {
        stubFor(get(urlPathEqualTo("/Qwen/Qwen3-0.6B/resolve/main/config.json")).willReturn(okJson(QWEN3_CONFIG_JSON)));

        huggingFaceClient.fetchFileAsJson("Qwen/Qwen3-0.6B", "config.json").blockingGet();

        WireMock.verify(
            1,
            WireMock.getRequestedFor(urlPathEqualTo("/Qwen/Qwen3-0.6B/resolve/main/config.json")).withoutHeader("Authorization")
        );
    }

    // -------------------------------------------------------------------------
    // downloadModelFile (existing coverage preserved)
    // -------------------------------------------------------------------------

    @Test
    @DisplayName("downloadModelFile: errors on 404 response")
    void shouldReturnErrorWhenDownloadFailsWith404() throws IOException {
        String modelName = "minuva/MiniLMv2-toxic-jigsaw-onnx";
        String fileName = "missing.json";

        stubFor(
            get(urlPathEqualTo("/minuva/MiniLMv2-toxic-jigsaw-onnx/resolve/main/missing.json"))
                .withQueryParam("download", equalTo("true"))
                .willReturn(aResponse().withStatus(404).withBody("Not found"))
        );

        Path tempFile = Files.createTempFile("hf-test-", ".json");
        AsyncFile asyncFile = Vertx
            .vertx()
            .fileSystem()
            .rxOpen(tempFile.toString(), new OpenOptions().setCreate(true).setWrite(true).setTruncateExisting(true))
            .blockingGet();

        Throwable thrown = catchThrowable(() -> huggingFaceClient.downloadModelFile(modelName, fileName, asyncFile).blockingAwait());

        assertThat(thrown)
            .isInstanceOf(IllegalStateException.class)
            .hasMessageContaining("Failed to download file: missing.json, status: 404, message: Not Found, body: null");

        Files.deleteIfExists(tempFile);
    }
}
