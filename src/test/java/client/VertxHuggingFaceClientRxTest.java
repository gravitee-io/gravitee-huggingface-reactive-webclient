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

import com.github.tomakehurst.wiremock.junit5.WireMockTest;
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
import org.junit.jupiter.api.Test;

@WireMockTest(httpPort = 8080)
class VertxHuggingFaceClientRxTest {

    VertxHuggingFaceClientRx huggingFaceClient;

    @BeforeEach
    void setUp() {
        var client = WebClient.create(
            Vertx.vertx(),
            new WebClientOptions().setDefaultHost("localhost").setDefaultPort(8080).setSsl(false).setLogActivity(true)
        );
        this.huggingFaceClient = new VertxHuggingFaceClientRx(client);
    }

    @Test
    void shouldReturnListOfModelFiles() {
        // given
        stubFor(
            get(urlEqualTo("/api/models/minuva/MiniLMv2-toxic-jigsaw-onnx"))
                .willReturn(
                    okJson(
                        """
            {
              "siblings": [
                { "rfilename": "config.json" },
                { "rfilename": "tokenizer.json" },
                { "rfilename": "model_optimized_quantized.onnx" }
              ]
            }
        """
                    )
                )
        );

        // when
        List<String> files = huggingFaceClient.listModelFiles("minuva/MiniLMv2-toxic-jigsaw-onnx").toList().blockingGet();

        // then
        assertThat(files).containsExactlyInAnyOrder("config.json", "tokenizer.json", "model_optimized_quantized.onnx");
    }

    @Test
    void shouldCompleteWithoutErrorWhenApiReturns404() {
        // given
        stubFor(get(urlEqualTo("/api/models/missing-model")).willReturn(aResponse().withStatus(404)));

        // when
        List<String> result = huggingFaceClient.listModelFiles("missing-model").toList().blockingGet();

        // then
        assertThat(result).isEmpty();
    }

    @Test
    void shouldDownloadModelFileSuccessfully() throws IOException {
        // given
        String modelName = "minuva/MiniLMv2-toxic-jigsaw-onnx";
        String fileName = "config.json";
        String fileContent =
            """
                {
                  "_name_or_path": "../output/MiniLM-L6-toxic-all-labels-opt",
                  "architectures": [
                    "BertForSequenceClassification"
                  ],
                  "attention_probs_dropout_prob": 0.1,
                  "classifier_dropout": null,
                  "hidden_act": "gelu",
                  "hidden_dropout_prob": 0.1,
                  "hidden_size": 384,
                  "id2label": {
                    "0": "toxic",
                    "1": "severe_toxic",
                    "2": "obscene",
                    "3": "threat",
                    "4": "insult",
                    "5": "identity_hate"
                  },
                  "initializer_range": 0.02,
                  "intermediate_size": 1536,
                  "label2id": {
                    "identity_hate": "5",
                    "insult": "4",
                    "obscene": "2",
                    "severe_toxic": "1",
                    "threat": "3",
                    "toxic": "0"
                  },
                  "layer_norm_eps": 1e-12,
                  "max_position_embeddings": 512,
                  "model_type": "bert",
                  "num_attention_heads": 12,
                  "num_hidden_layers": 6,
                  "pad_token_id": 0,
                  "position_embedding_type": "absolute",
                  "problem_type": "multi_label_classification",
                  "torch_dtype": "float32",
                  "transformers_version": "4.30.0",
                  "type_vocab_size": 2,
                  "use_cache": true,
                  "vocab_size": 30522
                }
                """;

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

    @Test
    void shouldReturnErrorWhenDownloadFailsWith404() throws IOException {
        // given
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

        // when
        Throwable thrown = catchThrowable(() -> huggingFaceClient.downloadModelFile(modelName, fileName, asyncFile).blockingAwait());

        // then
        assertThat(thrown)
            .isInstanceOf(IllegalStateException.class)
            .hasMessageContaining("Failed to download file: missing.json, status: 404, message: Not Found, body: null");

        Files.deleteIfExists(tempFile);
    }
}
