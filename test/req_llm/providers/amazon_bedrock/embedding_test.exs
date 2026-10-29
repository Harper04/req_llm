defmodule ReqLLM.Providers.AmazonBedrock.EmbeddingTest do
  use ExUnit.Case, async: true

  alias ReqLLM.Providers.AmazonBedrock

  describe "prepare_request/4 for embedding" do
    setup do
      System.put_env("AWS_ACCESS_KEY_ID", "AKIATEST")
      System.put_env("AWS_SECRET_ACCESS_KEY", "secretTEST")
      System.put_env("AWS_REGION", "us-east-1")
      :ok
    end

    test "builds embedding request for Cohere model" do
      {:ok, model} = ReqLLM.model("amazon-bedrock:cohere.embed-english-v3")
      text = "Hello, world!"

      opts = [
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST",
        region: "us-east-1"
      ]

      {:ok, request} = AmazonBedrock.prepare_request(:embedding, model, text, opts)

      assert %Req.Request{} = request
      assert request.url.path =~ "/model/cohere.embed-english-v3/invoke"
    end

    test "preserves inference profile prefix for Cohere embedding model" do
      output =
        ExUnit.CaptureIO.capture_io(:stderr, fn ->
          send(self(), {:model_result, ReqLLM.model("amazon-bedrock:global.cohere.embed-v4:0")})
        end)

      assert output == ""
      assert_received {:model_result, {:ok, model}}
      assert model.id == "cohere.embed-v4"
      assert is_number(model.cost.input)

      text = "Hello, world!"

      opts = [
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST",
        region: "us-east-1"
      ]

      {:ok, request} = AmazonBedrock.prepare_request(:embedding, model, text, opts)

      assert model.provider_model_id == "global.cohere.embed-v4:0"
      assert request.url.path == "/model/global.cohere.embed-v4:0/invoke"
    end

    test "includes text in Cohere format" do
      {:ok, model} = ReqLLM.model("amazon-bedrock:cohere.embed-english-v3")
      text = "Test embedding text"

      opts = [
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST"
      ]

      {:ok, request} = AmazonBedrock.prepare_request(:embedding, model, text, opts)

      body = ReqLLM.Test.Helpers.json_body(request)
      assert body["texts"] == ["Test embedding text"]
      assert body["input_type"] == "search_document"
      assert body["embedding_types"] == ["float"]
    end

    test "supports batch text input" do
      {:ok, model} = ReqLLM.model("amazon-bedrock:cohere.embed-english-v3")
      texts = ["First text", "Second text", "Third text"]

      opts = [
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST"
      ]

      {:ok, request} = AmazonBedrock.prepare_request(:embedding, model, texts, opts)

      body = ReqLLM.Test.Helpers.json_body(request)
      assert body["texts"] == ["First text", "Second text", "Third text"]
    end

    test "supports custom input_type via provider_options" do
      {:ok, model} = ReqLLM.model("amazon-bedrock:cohere.embed-english-v3")
      text = "Query text"

      opts = [
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST",
        provider_options: [input_type: "search_query"]
      ]

      {:ok, request} = AmazonBedrock.prepare_request(:embedding, model, text, opts)

      body = ReqLLM.Test.Helpers.json_body(request)
      assert body["input_type"] == "search_query"
    end

    test "supports embedding_types option" do
      {:ok, model} = ReqLLM.model("amazon-bedrock:cohere.embed-english-v3")
      text = "Test text"

      opts = [
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST",
        provider_options: [embedding_types: ["float", "int8"]]
      ]

      {:ok, request} = AmazonBedrock.prepare_request(:embedding, model, text, opts)

      body = ReqLLM.Test.Helpers.json_body(request)
      assert body["embedding_types"] == ["float", "int8"]
    end

    test "supports truncate option" do
      {:ok, model} = ReqLLM.model("amazon-bedrock:cohere.embed-english-v3")
      text = "Test text"

      opts = [
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST",
        provider_options: [truncate: "RIGHT"]
      ]

      {:ok, request} = AmazonBedrock.prepare_request(:embedding, model, text, opts)

      body = ReqLLM.Test.Helpers.json_body(request)
      assert body["truncate"] == "RIGHT"
    end
  end

  describe "attach_embedding/3" do
    setup do
      {:ok, model} = ReqLLM.model("amazon-bedrock:cohere.embed-english-v3")

      opts = [
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST",
        region: "us-east-1",
        text: "Hello, world!"
      ]

      {:ok, model: model, opts: opts}
    end

    test "attaches AWS SigV4 signing", %{model: model, opts: opts} do
      request = Req.new(url: "/model/cohere.embed-english-v3/invoke", method: :post)

      attached = AmazonBedrock.attach_embedding(request, model, opts)

      assert attached.request_steps[:aws_sigv4] != nil
    end

    test "routes the request to the configured Finch pool", %{model: model, opts: opts} do
      request = Req.new(url: "/model/cohere.embed-english-v3/invoke", method: :post)

      attached = AmazonBedrock.attach_embedding(request, model, opts)

      assert attached.options[:finch] == [name: ReqLLM.Application.finch_name()]
    end

    test "sets content-type header", %{model: model, opts: opts} do
      request = Req.new(url: "/model/cohere.embed-english-v3/invoke", method: :post)

      attached = AmazonBedrock.attach_embedding(request, model, opts)

      headers_map = Map.new(attached.headers)
      assert headers_map["content-type"] == ["application/json"]
    end

    test "configures base URL with region", %{model: model, opts: opts} do
      request = Req.new(url: "/model/cohere.embed-english-v3/invoke", method: :post)

      attached = AmazonBedrock.attach_embedding(request, model, opts)

      assert attached.options[:base_url] == "https://bedrock-runtime.us-east-1.amazonaws.com"
    end

    test "uses custom region", %{model: model, opts: opts} do
      request = Req.new(url: "/model/cohere.embed-english-v3/invoke", method: :post)
      custom_opts = Keyword.put(opts, :region, "eu-west-1")

      attached = AmazonBedrock.attach_embedding(request, model, custom_opts)

      assert attached.options[:base_url] == "https://bedrock-runtime.eu-west-1.amazonaws.com"
    end

    test "sets model family in options", %{model: model, opts: opts} do
      request = Req.new(url: "/model/cohere.embed-english-v3/invoke", method: :post)

      attached = AmazonBedrock.attach_embedding(request, model, opts)

      assert attached.options[:model_family] == "cohere"
    end

    test "attaches decode_embedding response step", %{model: model, opts: opts} do
      request = Req.new(url: "/model/cohere.embed-english-v3/invoke", method: :post)

      attached = AmazonBedrock.attach_embedding(request, model, opts)

      assert attached.response_steps[:llm_decode_embedding] != nil
    end

    test "attaches usage step", %{model: model, opts: opts} do
      request = Req.new(url: "/model/cohere.embed-english-v3/invoke", method: :post)

      attached = AmazonBedrock.attach_embedding(request, model, opts)

      assert attached.response_steps[:llm_usage] != nil
    end
  end

  describe "Cohere embedding formatter" do
    alias ReqLLM.Providers.AmazonBedrock.Cohere

    test "format_embedding_request builds correct request body" do
      {:ok, body} = Cohere.format_embedding_request("cohere.embed-english-v3", "hello", [])

      assert body["texts"] == ["hello"]
      assert body["input_type"] == "search_document"
      assert body["embedding_types"] == ["float"]
    end

    test "format_embedding_request handles list of texts" do
      texts = ["first", "second"]
      {:ok, body} = Cohere.format_embedding_request("cohere.embed-english-v3", texts, [])

      assert body["texts"] == ["first", "second"]
    end

    test "format_embedding_request supports provider_options" do
      opts = [
        provider_options: [
          input_type: "search_query",
          embedding_types: ["float", "int8"],
          truncate: "LEFT"
        ],
        dimensions: 256
      ]

      {:ok, body} = Cohere.format_embedding_request("cohere.embed-v4", "query", opts)

      assert body["input_type"] == "search_query"
      assert body["embedding_types"] == ["float", "int8"]
      assert body["truncate"] == "LEFT"
      assert body["output_dimension"] == 256
    end

    test "format_embedding_request validates input_type" do
      invalid_opts = [provider_options: [input_type: "invalid_type"]]

      {:error, error} =
        Cohere.format_embedding_request("cohere.embed-english-v3", "text", invalid_opts)

      assert %ReqLLM.Error.Validation.Error{} = error
      assert error.tag == :invalid_embedding_request
    end

    test "parse_embedding_response normalizes Cohere response" do
      cohere_response = %{
        "embeddings" => %{
          "float" => [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        }
      }

      {:ok, normalized} = Cohere.parse_embedding_response(cohere_response)

      assert normalized["data"] == [
               %{"index" => 0, "embedding" => [0.1, 0.2, 0.3]},
               %{"index" => 1, "embedding" => [0.4, 0.5, 0.6]}
             ]
    end

    test "parse_embedding_response handles direct embeddings list" do
      cohere_response = %{
        "embeddings" => [[0.1, 0.2], [0.3, 0.4]]
      }

      {:ok, normalized} = Cohere.parse_embedding_response(cohere_response)

      assert normalized["data"] == [
               %{"index" => 0, "embedding" => [0.1, 0.2]},
               %{"index" => 1, "embedding" => [0.3, 0.4]}
             ]
    end

    test "parse_embedding_response returns error for invalid response" do
      {:error, error} = Cohere.parse_embedding_response("not a map")

      assert %ReqLLM.Error.API.Response{} = error
    end
  end
end
