defmodule ReqLLM.Providers.AmazonBedrock.StructuredOutputTest do
  use ExUnit.Case, async: true

  import ReqLLM.Test.AmazonBedrockHelpers, only: [bedrock_opts: 1, sparse_adaptive_opus: 0]

  alias ReqLLM.{Context, Providers.AmazonBedrock}

  describe "prepared object streaming" do
    for use_converse <- [true, false], mode <- [:auto, :json_schema] do
      test "builds a stream from prepared options with Converse=#{use_converse}, mode=#{mode}" do
        {:ok, model} =
          ReqLLM.model("amazon_bedrock:anthropic.claude-haiku-4-5-20251001-v1:0")

        {:ok, schema} = ReqLLM.Schema.compile(name: [type: :string, required: true])
        context = Context.new([Context.user("Describe a city")])

        opts = [
          compiled_schema: schema,
          provider_options: [
            access_key_id: "AKIATEST",
            secret_access_key: "secretTEST",
            region: "eu-west-1",
            use_converse: unquote(use_converse),
            anthropic_structured_output_mode: unquote(mode)
          ]
        ]

        {:ok, prepared} = AmazonBedrock.prepare_request(:object, model, context, opts)

        stream_opts =
          opts
          |> Keyword.merge(Map.to_list(prepared.options))
          |> Keyword.put(:stream, true)

        assert {:ok, stream} =
                 AmazonBedrock.attach_stream(
                   model,
                   prepared.options[:context],
                   stream_opts,
                   ReqLLM.Finch
                 )

        assert stream.host == "bedrock-runtime.eu-west-1.amazonaws.com"
        assert Map.new(stream.headers)["authorization"] =~ "AWS4-HMAC-SHA256"
        assert Jason.decode!(stream.body) == Jason.decode!(prepared.body)

        if unquote(use_converse) do
          assert stream.path =~ "/converse-stream"
        else
          assert stream.path =~ "/invoke-with-response-stream"
        end
      end
    end
  end

  describe "anthropic_structured_output_mode: :json_schema" do
    setup do
      {:ok, claude} = ReqLLM.model("amazon_bedrock:anthropic.claude-haiku-4-5-20251001-v1:0")

      {:ok, compiled_schema} =
        ReqLLM.Schema.compile(
          type: [type: :string, required: true],
          text: [type: :string, required: true]
        )

      {:ok,
       claude: claude,
       compiled_schema: compiled_schema,
       context: Context.new([Context.user("Describe a city")])}
    end

    test "sends the schema as outputConfig.textFormat on Converse, buffered and streamed", %{
      claude: model,
      compiled_schema: compiled_schema,
      context: context
    } do
      opts = json_schema_opts(compiled_schema, use_converse: true)
      {:ok, request} = AmazonBedrock.prepare_request(:object, model, context, opts)
      {:ok, stream} = AmazonBedrock.attach_stream(model, context, opts, ReqLLM.Finch)

      for body <- [request.body, stream.body] do
        body = Jason.decode!(body, objects: :ordered_objects)
        text_format = body["outputConfig"]["textFormat"]

        schema =
          Jason.decode!(text_format["structure"]["jsonSchema"]["schema"],
            objects: :ordered_objects
          )

        assert text_format["type"] == "json_schema"
        assert Enum.map(schema["properties"], &elem(&1, 0)) == ["type", "text"]
        assert schema["required"] == ["type", "text"]
        assert schema["additionalProperties"] == false
        assert body["toolConfig"] == nil
      end
    end

    test "sends the schema as output_config.format on InvokeModel, buffered and streamed", %{
      claude: model,
      compiled_schema: compiled_schema,
      context: context
    } do
      opts = json_schema_opts(compiled_schema, [])
      {:ok, request} = AmazonBedrock.prepare_request(:object, model, context, opts)
      {:ok, stream} = AmazonBedrock.attach_stream(model, context, opts, ReqLLM.Finch)

      assert request.url.path =~ "/invoke"

      for body <- [request.body, stream.body] do
        body = Jason.decode!(body, objects: :ordered_objects)
        format = body["output_config"]["format"]

        assert format["type"] == "json_schema"
        assert Enum.map(format["schema"]["properties"], &elem(&1, 0)) == ["type", "text"]
        assert body["tools"] == nil
      end
    end

    test "reads the object from the message text on Converse and InvokeModel", %{
      claude: model,
      compiled_schema: compiled_schema,
      context: context
    } do
      answer = ~s({"type":"text","text":"Hello"})

      converse = %{
        "output" => %{"message" => %{"role" => "assistant", "content" => [%{"text" => answer}]}},
        "stopReason" => "end_turn",
        "usage" => %{"inputTokens" => 10, "outputTokens" => 5}
      }

      invoke = %{
        "id" => "msg_1",
        "type" => "message",
        "role" => "assistant",
        "model" => "claude-haiku-4-5",
        "content" => [%{"type" => "text", "text" => answer}],
        "stop_reason" => "end_turn",
        "usage" => %{"input_tokens" => 10, "output_tokens" => 5}
      }

      for {provider_options, payload} <- [{[use_converse: true], converse}, {[], invoke}] do
        opts = json_schema_opts(compiled_schema, provider_options)
        {:ok, request} = AmazonBedrock.prepare_request(:object, model, context, opts)

        {_request, response} =
          AmazonBedrock.decode_response({request, %Req.Response{status: 200, body: payload}})

        assert response.body.object == %{"type" => "text", "text" => "Hello"}
      end
    end

    test "keeps the thinking and effort fields that the structured_output tool drops", %{
      compiled_schema: compiled_schema,
      context: context
    } do
      thinking = %{"display" => "summarized", "type" => "adaptive"}

      body = fn mode, provider_options ->
        opts =
          bedrock_opts(
            operation: :object,
            compiled_schema: compiled_schema,
            reasoning_effort: :high,
            provider_options: [anthropic_structured_output_mode: mode] ++ provider_options
          )

        {:ok, request} =
          AmazonBedrock.prepare_request(:object, sparse_adaptive_opus(), context, opts)

        ReqLLM.Test.Helpers.json_body(request)
      end

      invoke = body.(:json_schema, [])
      assert invoke["thinking"] == thinking
      assert invoke["output_config"]["effort"] == "high"
      assert invoke["output_config"]["format"]["type"] == "json_schema"

      converse = body.(:json_schema, use_converse: true)
      assert converse["additionalModelRequestFields"]["thinking"] == thinking
      assert converse["additionalModelRequestFields"]["output_config"]["effort"] == "high"
      assert converse["outputConfig"]["textFormat"]["type"] == "json_schema"

      auto = body.(:auto, [])
      refute Map.has_key?(auto, "thinking")
      refute Map.has_key?(auto, "output_config")

      auto_converse = body.(:auto, use_converse: true)
      refute Map.has_key?(auto_converse["additionalModelRequestFields"] || %{}, "thinking")
    end

    test "keeps the structured_output tool for models other than Claude", %{
      compiled_schema: compiled_schema,
      context: context
    } do
      {:ok, nova} = ReqLLM.model("amazon_bedrock:amazon.nova-lite-v1:0")

      {:ok, request} =
        AmazonBedrock.prepare_request(
          :object,
          nova,
          context,
          json_schema_opts(compiled_schema, [])
        )

      body = ReqLLM.Test.Helpers.json_body(request)
      assert [%{"toolSpec" => %{"name" => "structured_output"}}] = body["toolConfig"]["tools"]
      refute Map.has_key?(body, "outputConfig")
    end

    test "sends the schema as output_config.format on the bedrock-mantle Messages API, buffered and streamed",
         %{claude: model, compiled_schema: compiled_schema, context: context} do
      opts = json_schema_opts(compiled_schema, endpoint: :mantle)
      {:ok, request} = AmazonBedrock.prepare_request(:object, model, context, opts)
      {:ok, stream} = AmazonBedrock.attach_stream(model, context, opts, ReqLLM.Finch)

      assert request.url.path == "/anthropic/v1/messages"

      for body <- [request.body, stream.body] do
        body = Jason.decode!(body, objects: :ordered_objects)
        format = body["output_config"]["format"]

        assert format["type"] == "json_schema"
        assert Enum.map(format["schema"]["properties"], &elem(&1, 0)) == ["type", "text"]
        assert body["tools"] == nil
      end
    end

    test "is refused with cited documents on Converse, the only API that sends citations", %{
      claude: model,
      compiled_schema: compiled_schema
    } do
      context =
        Context.new([
          Context.user([
            ReqLLM.Message.ContentPart.text("Summarize this document"),
            ReqLLM.Message.ContentPart.file("%PDF-1.4", "doc.pdf", "application/pdf", %{
              citations: true
            })
          ])
        ])

      converse = json_schema_opts(compiled_schema, use_converse: true)

      assert_raise ReqLLM.Error.Invalid.Parameter, ~r/citations/, fn ->
        AmazonBedrock.prepare_request(:object, model, context, converse)
      end

      {:ok, request} =
        AmazonBedrock.prepare_request(
          :object,
          model,
          context,
          json_schema_opts(compiled_schema, [])
        )

      body = ReqLLM.Test.Helpers.json_body(request)
      [_text, document] = hd(body["messages"])["content"]

      assert document["type"] == "document"
      refute Map.has_key?(document, "citations")
      assert body["output_config"]["format"]["type"] == "json_schema"
    end

    test "defines the tools a Converse history uses, which Bedrock requires", %{
      claude: model,
      compiled_schema: compiled_schema
    } do
      context = %Context{
        messages: [
          Context.user("Find city c1"),
          %ReqLLM.Message{
            role: :assistant,
            content: [],
            tool_calls: [ReqLLM.ToolCall.new("call_1", "lookup", ~s({"id":"c1"}))]
          },
          Context.tool_result("call_1", "City c1: Lyon"),
          Context.user("Describe it")
        ]
      }

      opts = json_schema_opts(compiled_schema, use_converse: true)
      {:ok, request} = AmazonBedrock.prepare_request(:object, model, context, opts)
      body = ReqLLM.Test.Helpers.json_body(request)

      assert [%{"toolSpec" => %{"name" => "lookup"}}] = body["toolConfig"]["tools"]
      refute Map.has_key?(body["toolConfig"], "toolChoice")
      assert body["outputConfig"]["textFormat"]["type"] == "json_schema"
    end

    test "sends the caller's tools next to the schema on Converse and InvokeModel", %{
      claude: model,
      compiled_schema: compiled_schema,
      context: context
    } do
      {:ok, lookup} =
        ReqLLM.Tool.new(
          name: "lookup",
          description: "Look up a city",
          parameter_schema: [id: [type: :string]],
          callback: fn _args -> {:ok, "found"} end
        )

      converse_opts =
        Keyword.put(json_schema_opts(compiled_schema, use_converse: true), :tools, [lookup])

      invoke_opts =
        Keyword.put(json_schema_opts(compiled_schema, use_converse: false), :tools, [lookup])

      {:ok, converse} = AmazonBedrock.prepare_request(:object, model, context, converse_opts)
      {:ok, invoke} = AmazonBedrock.prepare_request(:object, model, context, invoke_opts)

      converse = ReqLLM.Test.Helpers.json_body(converse)
      invoke = ReqLLM.Test.Helpers.json_body(invoke)

      assert [%{"toolSpec" => %{"name" => "lookup"}}] = converse["toolConfig"]["tools"]
      assert converse["outputConfig"]["textFormat"]["type"] == "json_schema"
      assert [%{"name" => "lookup"}] = invoke["tools"]
      assert invoke["output_config"]["format"]["type"] == "json_schema"
    end

    test "holds nested objects to the strict requirements on Converse and InvokeModel", %{
      claude: model,
      context: context
    } do
      {:ok, compiled_schema} =
        ReqLLM.Schema.compile(%{
          "type" => "object",
          "properties" => %{
            "city" => %{
              "type" => "object",
              "properties" => %{"name" => %{"type" => "string"}},
              "additionalProperties" => true
            },
            "stops" => %{
              "type" => "array",
              "items" => %{"type" => "object", "properties" => %{"id" => %{"type" => "string"}}}
            },
            "home" => %{"$ref" => "#/$defs/place"}
          },
          "$defs" => %{
            "place" => %{
              "type" => "object",
              "properties" => %{"label" => %{"type" => "string"}},
              "additionalProperties" => true
            }
          }
        })

      for provider_options <- [[use_converse: true], []] do
        opts = json_schema_opts(compiled_schema, provider_options)
        {:ok, request} = AmazonBedrock.prepare_request(:object, model, context, opts)
        schema = request |> ReqLLM.Test.Helpers.json_body() |> sent_schema()

        for object <- [
              schema["properties"]["city"],
              schema["properties"]["stops"]["items"],
              schema["$defs"]["place"]
            ] do
          assert object["additionalProperties"] == false
          assert object["required"] == Map.keys(object["properties"])
        end
      end
    end

    defp sent_schema(%{"outputConfig" => %{"textFormat" => text_format}}),
      do: Jason.decode!(text_format["structure"]["jsonSchema"]["schema"])

    defp sent_schema(%{"output_config" => %{"format" => %{"schema" => schema}}}), do: schema

    defp json_schema_opts(compiled_schema, provider_options) do
      bedrock_opts(
        operation: :object,
        compiled_schema: compiled_schema,
        provider_options: [anthropic_structured_output_mode: :json_schema] ++ provider_options
      )
    end
  end
end
