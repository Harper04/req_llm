defmodule ReqLLM.Providers.AmazonBedrock.StreamingTest do
  use ExUnit.Case, async: true

  import ReqLLM.Test.AmazonBedrockHelpers, only: [build_aws_event_stream_message: 1]

  alias ReqLLM.{Context, Providers.AmazonBedrock}

  describe "parse_stream_protocol/2" do
    test "parses AWS Event Stream binary data" do
      # Create a simple AWS Event Stream message
      payload = Jason.encode!(%{"type" => "chunk", "data" => "test"})
      binary = build_aws_event_stream_message(payload)

      assert {:ok, events, rest} = AmazonBedrock.parse_stream_protocol(binary, <<>>)
      refute Enum.empty?(events)
      assert rest == <<>>
    end

    test "handles incomplete data" do
      # Incomplete prelude
      partial = <<0, 0, 0, 100>>

      assert {:incomplete, buffer} = AmazonBedrock.parse_stream_protocol(partial, <<>>)
      assert buffer == partial
    end

    test "accumulates with buffer" do
      # Split a message across two chunks
      payload = Jason.encode!(%{"test" => "data"})
      full_message = build_aws_event_stream_message(payload)

      split = div(byte_size(full_message), 2)
      part1 = binary_part(full_message, 0, split)
      part2 = binary_part(full_message, split, byte_size(full_message) - split)

      # First chunk should be incomplete
      assert {:incomplete, buffer} = AmazonBedrock.parse_stream_protocol(part1, <<>>)

      # Second chunk should complete the message
      assert {:ok, events, <<>>} = AmazonBedrock.parse_stream_protocol(part2, buffer)
      assert length(events) == 1
    end
  end

  describe "unwrap_stream_chunk/1" do
    alias ReqLLM.Providers.AmazonBedrock.Response

    test "unwraps AWS SDK format with chunk wrapper" do
      event = %{"type" => "content_block_delta", "delta" => %{"text" => "hello"}}
      encoded = Base.encode64(Jason.encode!(event))
      chunk = %{"chunk" => %{"bytes" => encoded}}

      assert {:ok, unwrapped} = Response.unwrap_stream_chunk(chunk)
      assert unwrapped == event
    end

    test "unwraps direct bytes format" do
      event = %{"type" => "message_start", "message" => %{"id" => "msg_123"}}
      encoded = Base.encode64(Jason.encode!(event))
      chunk = %{"bytes" => encoded}

      assert {:ok, unwrapped} = Response.unwrap_stream_chunk(chunk)
      assert unwrapped == event
    end

    test "passes through already decoded events" do
      event = %{"type" => "message_stop"}
      chunk = event

      assert {:ok, unwrapped} = Response.unwrap_stream_chunk(chunk)
      assert unwrapped == event
    end

    test "returns error for unknown format" do
      chunk = %{"unknown" => "format"}

      assert {:error, :unknown_chunk_format} = Response.unwrap_stream_chunk(chunk)
    end

    test "returns error for invalid base64" do
      chunk = %{"bytes" => "invalid-base64!!!"}

      assert {:error, {:unwrap_failed, _}} = Response.unwrap_stream_chunk(chunk)
    end

    test "returns error for invalid JSON" do
      encoded = Base.encode64("not valid json")
      chunk = %{"bytes" => encoded}

      assert {:error, {:unwrap_failed, _}} = Response.unwrap_stream_chunk(chunk)
    end
  end

  describe "decode_stream_event/2" do
    test "decodes native Anthropic events for inference profile models" do
      # Inference profile models (global.*) may use InvokeModel API which returns
      # native Anthropic event format, not Converse format
      model =
        LLMDB.Model.new!(%{
          id: "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
          provider: :amazon_bedrock
        })

      # Native Anthropic event from InvokeModel API
      event = %{
        "type" => "content_block_delta",
        "index" => 0,
        "delta" => %{"type" => "text_delta", "text" => "Hello"}
      }

      chunks = AmazonBedrock.decode_stream_event(event, model)
      assert [%ReqLLM.StreamChunk{type: :content, text: "Hello"}] = chunks
    end

    test "decodes Converse events for inference profile models" do
      model =
        LLMDB.Model.new!(%{
          id: "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
          provider: :amazon_bedrock
        })

      # Converse API event format
      event = %{
        "contentBlockDelta" => %{
          "contentBlockIndex" => 0,
          "delta" => %{"text" => "Hello"}
        }
      }

      chunks = AmazonBedrock.decode_stream_event(event, model)
      assert [%ReqLLM.StreamChunk{type: :content, text: "Hello"}] = chunks
    end

    test "decodes native events for non-inference-profile models" do
      model =
        LLMDB.Model.new!(%{
          id: "anthropic.claude-3-haiku-20240307-v1:0",
          provider: :amazon_bedrock
        })

      event = %{
        "type" => "content_block_delta",
        "index" => 0,
        "delta" => %{"type" => "text_delta", "text" => "World"}
      }

      chunks = AmazonBedrock.decode_stream_event(event, model)
      assert [%ReqLLM.StreamChunk{type: :content, text: "World"}] = chunks
    end
  end

  describe "decode_stream_event/3" do
    test "emits one reasoning detail for a native Anthropic thinking block" do
      model =
        LLMDB.Model.new!(%{
          id: "anthropic.claude-3-haiku-20240307-v1:0",
          provider: :amazon_bedrock
        })

      events = [
        %{
          "type" => "content_block_start",
          "index" => 0,
          "content_block" => %{"type" => "thinking", "thinking" => "", "signature" => ""}
        },
        %{
          "type" => "content_block_delta",
          "index" => 0,
          "delta" => %{"type" => "thinking_delta", "thinking" => "First part "}
        },
        %{
          "type" => "content_block_delta",
          "index" => 0,
          "delta" => %{"type" => "thinking_delta", "thinking" => "second part"}
        },
        %{
          "type" => "content_block_delta",
          "index" => 0,
          "delta" => %{"type" => "signature_delta", "signature" => "sig_test_123"}
        },
        %{"type" => "content_block_stop", "index" => 0}
      ]

      {chunks, _state} =
        Enum.reduce(events, {[], AmazonBedrock.init_stream_state(model)}, fn event,
                                                                             {acc, state} ->
          {event_chunks, next_state} = AmazonBedrock.decode_stream_event(event, model, state)
          {acc ++ event_chunks, next_state}
        end)

      assert Enum.filter(chunks, &(&1.type == :thinking)) |> Enum.map(& &1.text) == [
               "First part ",
               "second part"
             ]

      reasoning_chunks =
        Enum.filter(chunks, fn
          %ReqLLM.StreamChunk{type: :meta, metadata: %{reasoning_details: [_detail]}} -> true
          _ -> false
        end)

      assert [%ReqLLM.StreamChunk{metadata: %{reasoning_details: [detail]}}] = reasoning_chunks
      assert detail.text == "First part second part"
      assert detail.signature == "sig_test_123"
      assert detail.provider == :anthropic
      assert detail.format == "anthropic-thinking-v1"
      assert detail.index == 0
    end

    test "maps native guardrail interventions in both stream decoders" do
      model =
        LLMDB.Model.new!(%{id: "meta.llama3-8b-instruct-v1:0", provider: :amazon_bedrock})

      trace = %{"guardrail" => %{"actionReason" => "Guardrail blocked."}}

      event = %{
        "stop_reason" => "stop",
        "amazon-bedrock-guardrailAction" => "INTERVENED",
        "amazon-bedrock-trace" => trace
      }

      {stateful_chunks, _state} =
        AmazonBedrock.decode_stream_event(event, model, AmazonBedrock.init_stream_state(model))

      for chunks <- [stateful_chunks, AmazonBedrock.decode_stream_event(event, model)] do
        assert [%ReqLLM.StreamChunk{type: :meta, metadata: metadata}] =
                 Enum.filter(chunks, &match?(%{metadata: %{provider_meta: _}}, &1))

        assert metadata.finish_reason == :content_filter

        assert metadata.provider_meta == %{
                 "amazon-bedrock-guardrailAction" => "INTERVENED",
                 "amazon-bedrock-trace" => trace
               }
      end
    end

    test "keeps the finish reason when the guardrail did not intervene on message_stop" do
      model =
        LLMDB.Model.new!(%{
          id: "anthropic.claude-3-haiku-20240307-v1:0",
          provider: :amazon_bedrock
        })

      event = %{"type" => "message_stop", "amazon-bedrock-guardrailAction" => "NONE"}

      {chunks, _state} =
        AmazonBedrock.decode_stream_event(event, model, AmazonBedrock.init_stream_state(model))

      assert [%ReqLLM.StreamChunk{metadata: metadata}] =
               Enum.filter(chunks, &match?(%{metadata: %{provider_meta: _}}, &1))

      refute Map.has_key?(metadata, :finish_reason)
      assert metadata.provider_meta == %{"amazon-bedrock-guardrailAction" => "NONE"}
    end

    test "native Anthropic streamed reasoning round-trips into a single thinking block" do
      model =
        LLMDB.Model.new!(%{
          id: "anthropic.claude-3-haiku-20240307-v1:0",
          provider: :amazon_bedrock
        })

      context = %ReqLLM.Context{messages: []}

      chunks = [
        ReqLLM.StreamChunk.thinking("First part "),
        ReqLLM.StreamChunk.thinking("second part"),
        ReqLLM.StreamChunk.meta(%{
          reasoning_details: [
            %ReqLLM.Message.ReasoningDetails{
              text: "First part second part",
              signature: "sig_test_123",
              encrypted?: true,
              provider: :anthropic,
              format: "anthropic-thinking-v1",
              index: 0,
              provider_data: %{"type" => "thinking"}
            }
          ]
        }),
        ReqLLM.StreamChunk.text("Answer: 42")
      ]

      builder = ReqLLM.Provider.ResponseBuilder.for_model(model)

      {:ok, response} =
        builder.build_response(chunks, %{finish_reason: :stop}, context: context, model: model)

      assert [detail] = response.message.reasoning_details
      assert detail.text == "First part second part"
      assert detail.signature == "sig_test_123"
      assert detail.provider == :anthropic

      encoded =
        ReqLLM.Providers.AmazonBedrock.Anthropic.format_request(model.id, response.context,
          model: model.id
        )

      [assistant_message] = encoded[:messages]
      [thinking_block, text_block] = assistant_message[:content]

      assert thinking_block[:type] == "thinking"
      assert thinking_block[:thinking] == "First part second part"
      assert thinking_block[:signature] == "sig_test_123"
      assert text_block == %{type: "text", text: "Answer: 42"}
    end
  end

  describe "Converse reasoning stream state" do
    setup do
      {:ok, model} =
        ReqLLM.model(%{
          provider: :amazon_bedrock,
          id: "anthropic.claude-haiku-4-5-20251001-v1:0",
          capabilities: %{chat: true, reasoning: %{enabled: true}}
        })

      {:ok, model: model}
    end

    test "starts from nil state and flushes pending reasoning for Claude", %{model: model} do
      delta = %{
        "contentBlockDelta" => %{
          "contentBlockIndex" => 0,
          "delta" => %{"reasoningContent" => %{"text" => "thought"}}
        }
      }

      signature = %{
        "contentBlockDelta" => %{
          "contentBlockIndex" => 0,
          "delta" => %{"reasoningContent" => %{"signature" => "sig"}}
        }
      }

      {[%ReqLLM.StreamChunk{type: :thinking}], state} =
        AmazonBedrock.decode_stream_event(delta, model, nil)

      {[], state} = AmazonBedrock.decode_stream_event(signature, model, state)
      {chunks, _state} = AmazonBedrock.flush_stream_state(model, state)

      assert [
               %ReqLLM.StreamChunk{
                 type: :meta,
                 metadata: %{
                   reasoning_details: [
                     %ReqLLM.Message.ReasoningDetails{text: "thought", signature: "sig"}
                   ]
                 }
               }
             ] = chunks
    end
  end

  describe "attach_stream/4" do
    setup do
      {:ok, model} = ReqLLM.model("amazon-bedrock:anthropic.claude-3-haiku-20240307-v1:0")
      context = Context.new([Context.user("Hello")])

      opts = [
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST",
        region: "us-east-1"
      ]

      {:ok, model: model, context: context, opts: opts}
    end

    test "builds Finch.Request for streaming", %{model: model, context: context, opts: opts} do
      assert {:ok, finch_request} =
               AmazonBedrock.attach_stream(model, context, opts, ReqLLM.Finch)

      assert %Finch.Request{} = finch_request
      assert finch_request.method == "POST"
      assert finch_request.path =~ "/model/"
      assert finch_request.path =~ "/invoke-with-response-stream"
    end

    test "includes proper headers", %{model: model, context: context, opts: opts} do
      assert {:ok, finch_request} =
               AmazonBedrock.attach_stream(model, context, opts, ReqLLM.Finch)

      headers_map = Map.new(finch_request.headers)
      assert headers_map["content-type"] == "application/json"
      assert headers_map["accept"] == "application/vnd.amazon.eventstream"
      assert Map.has_key?(headers_map, "authorization")
    end

    test "signs request with AWS SigV4", %{model: model, context: context, opts: opts} do
      assert {:ok, finch_request} =
               AmazonBedrock.attach_stream(model, context, opts, ReqLLM.Finch)

      # Check for AWS signature in authorization header
      auth_header = Enum.find(finch_request.headers, fn {k, _} -> k == "authorization" end)
      assert auth_header != nil
      {_, auth_value} = auth_header
      assert auth_value =~ "AWS4-HMAC-SHA256"
    end

    test "uses correct region", %{model: model, context: context, opts: opts} do
      custom_opts = Keyword.put(opts, :region, "eu-west-1")

      assert {:ok, finch_request} =
               AmazonBedrock.attach_stream(model, context, custom_opts, ReqLLM.Finch)

      assert finch_request.host =~ "eu-west-1"
    end

    test "includes session token in signed request", %{model: model, context: context, opts: opts} do
      session_token = "FwoGZXIvYXdzEBYaDHhBTEMPLESessionToken123"
      opts_with_token = Keyword.put(opts, :session_token, session_token)

      assert {:ok, finch_request} =
               AmazonBedrock.attach_stream(model, context, opts_with_token, ReqLLM.Finch)

      # Verify session token is included in headers
      headers_map = Map.new(finch_request.headers)
      assert headers_map["x-amz-security-token"] == session_token

      # Verify the request is still properly signed with the session token
      auth_header = headers_map["authorization"]
      assert auth_header != nil
      assert auth_header =~ "AWS4-HMAC-SHA256"
      # Session token should be included in the signed headers list
      assert auth_header =~ "x-amz-security-token"
    end
  end
end
