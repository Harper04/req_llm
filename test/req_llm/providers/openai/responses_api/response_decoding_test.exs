defmodule Provider.OpenAI.ResponsesAPI.ResponseDecodingTest do
  use ExUnit.Case, async: false

  import ReqLLM.Test.OpenAI.ResponsesAPIHelpers,
    only: [build_response: 2, build_response: 3, refusal_message_item: 2]

  alias ReqLLM.Providers.OpenAI.ResponsesAPI

  describe "decode_response/1" do
    test "decodes successful response with output_text field" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output_text" => "Hello world",
        "usage" => %{
          "input_tokens" => 10,
          "output_tokens" => 20
        }
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert %ReqLLM.Response{} = resp.body
      assert resp.body.id == "resp_123"
      assert resp.body.model == "gpt-5"
      assert resp.body.message.role == :assistant
      assert [part] = resp.body.message.content
      assert part.type == :text
      assert part.text == "Hello world"
      assert resp.body.usage.input_tokens == 10
      assert resp.body.usage.output_tokens == 20
      assert resp.body.usage.total_tokens == 30
    end

    test "decodes response with message segments" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output" => [
          %{
            "type" => "message",
            "content" => [
              %{"type" => "output_text", "text" => "Part 1 "},
              %{"type" => "output_text", "text" => "Part 2"}
            ]
          }
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert [part] = resp.body.message.content
      assert part.type == :text
      assert part.text == "Part 1 Part 2"
    end

    test "decodes a refusal as text with content_filter and the refusal retained" do
      refusal = "I'm sorry, I can't help with that."

      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "status" => "completed",
        "output" => [refusal_message_item("msg_1", refusal)],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert [%ReqLLM.Message.ContentPart{type: :text, text: ^refusal}] =
               resp.body.message.content

      assert resp.body.finish_reason == :content_filter
      assert resp.body.provider_meta["refusals"] == [refusal]
      assert ReqLLM.Response.refusals(resp.body) == [refusal]
    end

    test "a refusal takes precedence over tool calls for the finish reason" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "status" => "completed",
        "output" => [
          refusal_message_item("msg_1", "I can't help with that."),
          %{
            "type" => "function_call",
            "call_id" => "call_1",
            "name" => "lookup",
            "arguments" => ~s({"q":"docs"})
          }
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert resp.body.finish_reason == :content_filter
    end

    test "generate_object with a refusal skips object extraction" do
      {:ok, compiled_schema} = ReqLLM.Schema.compile(name: [type: :string, required: true])
      refusal = "I'm sorry, I can't help with that."

      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "status" => "completed",
        "output" => [refusal_message_item("msg_1", refusal)],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      req = %Req.Request{
        method: :post,
        url: URI.parse("https://api.openai.com/v1/responses"),
        headers: %{},
        body: {:json, %{}},
        options: %{id: "gpt-5", operation: :object, compiled_schema: compiled_schema}
      }

      resp = %Req.Response{status: 200, headers: %{}, body: response_body}

      {_req, decoded_resp} = ResponsesAPI.decode_response({req, resp})

      assert %ReqLLM.Response{object: nil, finish_reason: :content_filter} = decoded_resp.body
      refute Map.has_key?(decoded_resp.body.provider_meta, :object_parse_error)
      assert ReqLLM.Response.refusals(decoded_resp.body) == [refusal]
    end

    test "decodes a phased refusal as text with the phase" do
      refusal = "I can't help with that."

      item =
        "msg_1"
        |> refusal_message_item(refusal)
        |> Map.put("phase", "final_answer")

      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5.4",
        "status" => "completed",
        "output" => [item],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert [
               %ReqLLM.Message.ContentPart{
                 type: :text,
                 text: ^refusal,
                 metadata: %{phase: "final_answer"}
               }
             ] = resp.body.message.content

      assert resp.body.finish_reason == :content_filter
    end

    test "deduplicates identical commentary and final_answer message segments" do
      text = "Do you want a shortcut for all users or just the current user?"

      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5.4",
        "output" => [
          %{
            "type" => "message",
            "phase" => "commentary",
            "content" => [%{"type" => "output_text", "text" => text}]
          },
          %{
            "type" => "message",
            "phase" => "final_answer",
            "content" => [%{"type" => "output_text", "text" => text}]
          }
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert [part] = resp.body.message.content
      assert part.type == :text
      assert part.text == text
      assert part.metadata == %{phase: "final_answer"}
    end

    test "keeps distinct commentary and final_answer message segments as separate parts" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5.4",
        "output" => [
          %{
            "type" => "message",
            "phase" => "commentary",
            "content" => [%{"type" => "output_text", "text" => "Let me check that. "}]
          },
          %{
            "type" => "message",
            "phase" => "final_answer",
            "content" => [%{"type" => "output_text", "text" => "Here is the result."}]
          }
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert resp.body.message.content == [
               ReqLLM.Message.ContentPart.text("Let me check that. ", %{phase: "commentary"}),
               ReqLLM.Message.ContentPart.text("Here is the result.", %{phase: "final_answer"})
             ]

      assert ReqLLM.Response.text(resp.body) == "Let me check that. Here is the result."
    end

    test "preserves single-message phase metadata" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5.4",
        "output" => [
          %{
            "type" => "message",
            "phase" => "final_answer",
            "content" => [%{"type" => "output_text", "text" => "Root cause found."}]
          }
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert resp.body.message.metadata[:response_id] == "resp_123"
      assert resp.body.message.metadata[:phase] == "final_answer"
      refute Map.has_key?(resp.body.message.metadata, :phase_items)
    end

    test "preserves ordered phase_items for multi-segment phased output" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5.4",
        "output" => [
          %{
            "type" => "message",
            "phase" => "commentary",
            "content" => [%{"type" => "output_text", "text" => "Inspecting logs. "}]
          },
          %{
            "type" => "message",
            "phase" => "final_answer",
            "content" => [%{"type" => "output_text", "text" => "Cache race confirmed."}]
          }
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert resp.body.message.metadata[:response_id] == "resp_123"

      assert resp.body.message.metadata[:phase_items] == [
               %{
                 "phase" => "commentary",
                 "content" => [%{"type" => "output_text", "text" => "Inspecting logs. "}]
               },
               %{
                 "phase" => "final_answer",
                 "content" => [%{"type" => "output_text", "text" => "Cache race confirmed."}]
               }
             ]
    end

    test "decodes response with direct output_text segments" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output" => [
          %{"type" => "output_text", "text" => "Direct text"}
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert [part] = resp.body.message.content
      assert part.type == :text
      assert part.text == "Direct text"
    end

    test "aggregates text from multiple sources" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output_text" => "Top level ",
        "output" => [
          %{
            "type" => "message",
            "content" => [%{"type" => "output_text", "text" => "message "}]
          },
          %{"type" => "output_text", "text" => "direct"}
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert [part] = resp.body.message.content
      assert part.type == :text
      assert part.text == "Top level message direct"
    end

    test "decodes reasoning summary" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output" => [
          %{"type" => "reasoning", "summary" => "Thinking about this..."}
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert [thinking_part] = resp.body.message.content
      assert thinking_part.type == :thinking
      assert thinking_part.text == "Thinking about this..."
    end

    test "extracts reasoning summary text via Response.thinking/1" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output" => [
          %{
            "type" => "reasoning",
            "id" => "rs_123",
            "summary" => [
              %{"type" => "summary_text", "text" => "Determining if 143 is prime..."}
            ]
          }
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert ReqLLM.Response.thinking(resp.body) == "Determining if 143 is prime..."
    end

    test "decodes reasoning content" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output" => [
          %{
            "type" => "reasoning",
            "content" => [
              %{"text" => "Step 1 "},
              %{"text" => "Step 2"}
            ]
          }
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert [thinking_part] = resp.body.message.content
      assert thinking_part.type == :thinking
      assert thinking_part.text == "Step 1 Step 2"
    end

    test "aggregates reasoning from summary and content" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output" => [
          %{
            "type" => "reasoning",
            "summary" => "Summary ",
            "content" => [%{"text" => "details"}]
          }
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert [thinking_part] = resp.body.message.content
      assert thinking_part.type == :thinking
      assert thinking_part.text == "Summary details"
    end

    test "decodes tool calls" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output" => [
          %{
            "type" => "function_call",
            "call_id" => "call_abc",
            "name" => "get_weather",
            "arguments" => ~s({"location": "NYC"})
          }
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert [tool_call] = resp.body.message.tool_calls
      assert tool_call.id == "call_abc"
      assert tool_call.function.name == "get_weather"
      assert Jason.decode!(tool_call.function.arguments) == %{"location" => "NYC"}
    end

    test "decodes builtin tool calls as non-actionable final answers" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "status" => "completed",
        "output" => [
          %{
            "type" => "web_search_call",
            "id" => "ws_1",
            "status" => "completed",
            "action" => %{"type" => "search", "query" => "elixir"}
          }
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert resp.body.finish_reason == :stop
      assert [tool_call] = resp.body.message.tool_calls
      assert ReqLLM.ToolCall.builtin?(tool_call)
      assert tool_call.id == "ws_1"
      assert tool_call.function.name == "web_search_call"

      assert Jason.decode!(tool_call.function.arguments) == %{
               "action" => %{"type" => "search", "query" => "elixir"}
             }

      assert ReqLLM.ToolCall.metadata(tool_call) == %{status: "completed"}
      assert ReqLLM.Response.classify(resp.body).type == :final_answer
    end

    test "keeps a failed builtin call's status in its metadata, off the arguments" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "status" => "completed",
        "output" => [
          %{
            "type" => "web_search_call",
            "id" => "ws_1",
            "status" => "failed",
            "action" => %{"type" => "search", "query" => "elixir"}
          }
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert [tool_call] = resp.body.message.tool_calls
      assert ReqLLM.ToolCall.metadata(tool_call) == %{status: "failed"}
      refute Map.has_key?(Jason.decode!(tool_call.function.arguments), "status")
    end

    test "handles malformed tool call arguments" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output" => [
          %{
            "type" => "function_call",
            "call_id" => "call_abc",
            "name" => "get_weather",
            "arguments" => "invalid json"
          }
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert [tool_call] = resp.body.message.tool_calls
      assert tool_call.id == "call_abc"
      assert tool_call.function.name == "get_weather"
      assert tool_call.function.arguments == "invalid json"
    end

    test "normalizes usage with reasoning_tokens" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output_text" => "Hello",
        "usage" => %{
          "input_tokens" => 10,
          "output_tokens" => 20,
          "output_tokens_details" => %{
            "reasoning_tokens" => 5
          }
        }
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert resp.body.usage.input_tokens == 10
      assert resp.body.usage.output_tokens == 20
      assert resp.body.usage.total_tokens == 30
    end

    test "returns error for non-200 status" do
      {_req, result} = ResponsesAPI.decode_response(build_response(500, %{"error" => "boom"}))

      assert %ReqLLM.Error.API.Response{} = result
      assert result.status == 500
      assert result.reason == "OpenAI Responses API error"
    end

    test "handles missing usage gracefully" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output_text" => "Hello"
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert resp.body.usage.input_tokens == 0
      assert resp.body.usage.output_tokens == 0
      assert resp.body.usage.total_tokens == 0
    end

    test "handles malformed usage gracefully" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output_text" => "Hello",
        "usage" => "invalid"
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert resp.body.usage.input_tokens == 0
      assert resp.body.usage.output_tokens == 0
      assert resp.body.usage.total_tokens == 0
      assert resp.body.usage.usage_reported == %{input: false, output: false}
    end

    test "collects code_interpreter output items in provider_meta" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output" => [
          %{
            "type" => "message",
            "content" => [%{"type" => "output_text", "text" => "The result is 4."}]
          },
          %{
            "type" => "code_interpreter_call",
            "id" => "ci_xxx",
            "code" => "print(2+2)",
            "status" => "completed"
          },
          %{
            "type" => "code_interpreter_logs",
            "call_id" => "ci_xxx",
            "logs" => "4\n"
          },
          %{
            "type" => "code_interpreter_interpretation",
            "call_id" => "ci_xxx",
            "text" => "The result is 4."
          }
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert [part] = resp.body.message.content
      assert part.type == :text
      assert part.text == "The result is 4."
      assert resp.body.message.tool_calls == nil

      assert [call, logs, interpretation] =
               get_in(resp.body.provider_meta, ["code_interpreter", "items"])

      assert call["type"] == "code_interpreter_call"
      assert call["id"] == "ci_xxx"
      assert logs["type"] == "code_interpreter_logs"
      assert logs["logs"] == "4\n"
      assert interpretation["type"] == "code_interpreter_interpretation"
      assert interpretation["text"] == "The result is 4."
    end

    test "preserves code_interpreter_call outputs in provider_meta" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output" => [
          %{
            "type" => "code_interpreter_call",
            "id" => "ci_xxx",
            "code" => "print(2+2)",
            "status" => "completed",
            "outputs" => [
              %{"type" => "code_interpreter_logs", "logs" => "4\n"},
              %{"type" => "code_interpreter_interpretation", "text" => "Four."}
            ]
          }
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert [item] = get_in(resp.body.provider_meta, ["code_interpreter", "items"])
      assert item["type"] == "code_interpreter_call"
      assert [logs, interpretation] = item["outputs"]
      assert logs["type"] == "code_interpreter_logs"
      assert logs["logs"] == "4\n"
      assert interpretation["type"] == "code_interpreter_interpretation"
      assert interpretation["text"] == "Four."
    end

    test "counts code_interpreter usage in tool_usage" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output" => [
          %{
            "type" => "code_interpreter_call",
            "id" => "ci_xxx",
            "code" => "print(2+2)",
            "status" => "completed"
          }
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert resp.body.usage.tool_usage == %{code_interpreter: %{count: 1, unit: :call}}
    end

    test "keeps text and function calls separate from code_interpreter items" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output" => [
          %{
            "type" => "message",
            "content" => [%{"type" => "output_text", "text" => "The result is 4."}]
          },
          %{
            "type" => "function_call",
            "call_id" => "call_123",
            "name" => "get_weather",
            "arguments" => ~s({"location":"SF"})
          },
          %{
            "type" => "code_interpreter_call",
            "id" => "ci_xxx",
            "code" => "print(2+2)",
            "status" => "completed"
          }
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert [part] = resp.body.message.content
      assert part.type == :text
      assert part.text == "The result is 4."

      assert [%ReqLLM.ToolCall{function: %{name: "get_weather"}}] = resp.body.message.tool_calls
      assert resp.body.usage.tool_usage == %{code_interpreter: %{count: 1, unit: :call}}

      assert [call] = get_in(resp.body.provider_meta, ["code_interpreter", "items"])
      assert call["type"] == "code_interpreter_call"
      assert call["id"] == "ci_xxx"
    end

    test "appends message to request context" do
      msg = %ReqLLM.Message{
        role: :user,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Hello"}]
      }

      context = %ReqLLM.Context{messages: [msg]}

      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output_text" => "Hi",
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} =
        ResponsesAPI.decode_response(build_response(200, response_body, context: context))

      assert length(resp.body.context.messages) == 2
      assert Enum.at(resp.body.context.messages, 0).role == :user
      assert Enum.at(resp.body.context.messages, 1).role == :assistant
    end

    test "validates nested schemas with string keys converted to atoms" do
      result_schema =
        {:map,
         [
           id: [type: :pos_integer, required: true],
           reasoning: [type: :string, required: true],
           tags: [type: {:list, {:in, ~w[IT transport]}}, required: true]
         ]}

      schema = [
        results: [type: {:list, result_schema}, required: true]
      ]

      {:ok, compiled_schema} = ReqLLM.Schema.compile(schema)

      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output_text" =>
          ~s({"results": [{"id": 1, "reasoning": "Transport job", "tags": ["transport"]}]}),
        "usage" => %{"input_tokens" => 10, "output_tokens" => 20}
      }

      msg = %ReqLLM.Message{
        role: :user,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Tag these jobs"}]
      }

      context = %ReqLLM.Context{messages: [msg]}

      req = %Req.Request{
        method: :post,
        url: URI.parse("https://api.openai.com/v1/responses"),
        headers: %{},
        body: {:json, %{}},
        options: %{
          id: "gpt-5",
          context: context,
          operation: :object,
          compiled_schema: compiled_schema
        }
      }

      resp = %Req.Response{
        status: 200,
        headers: %{},
        body: response_body
      }

      {_req, decoded_resp} = ResponsesAPI.decode_response({req, resp})

      assert %ReqLLM.Response{} = decoded_resp.body
      assert decoded_resp.body.object != nil

      assert decoded_resp.body.object["results"] == [
               %{"id" => 1, "reasoning" => "Transport job", "tags" => ["transport"]}
             ]

      assert decoded_resp.body.provider_meta[:object_parse_error] == nil
    end

    test "extracts structured output from tool call when text is empty (gpt-5.4 style)" do
      schema = [name: [type: :string, required: true]]
      {:ok, compiled_schema} = ReqLLM.Schema.compile(schema)

      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5.4",
        "output" => [
          %{
            "type" => "function_call",
            "call_id" => "call_abc",
            "name" => "structured_output",
            "arguments" => ~s({"name": "Alice"})
          }
        ],
        "usage" => %{"input_tokens" => 10, "output_tokens" => 20}
      }

      req = %Req.Request{
        method: :post,
        url: URI.parse("https://api.openai.com/v1/responses"),
        headers: %{},
        body: {:json, %{}},
        options: %{
          id: "gpt-5",
          operation: :object,
          compiled_schema: compiled_schema
        }
      }

      resp = %Req.Response{status: 200, headers: %{}, body: response_body}

      {_req, decoded_resp} = ResponsesAPI.decode_response({req, resp})

      assert %ReqLLM.Response{} = decoded_resp.body
      assert decoded_resp.body.object == %{"name" => "Alice"}
      assert decoded_resp.body.provider_meta[:object_parse_error] == nil
    end

    test "validates structured output from tool calls against the compiled schema" do
      schema = [name: [type: :string, required: true], age: [type: :integer, required: true]]
      {:ok, compiled_schema} = ReqLLM.Schema.compile(schema)

      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5.4",
        "output" => [
          %{
            "type" => "function_call",
            "call_id" => "call_abc",
            "name" => "structured_output",
            "arguments" => ~s({"name": "Alice"})
          }
        ],
        "usage" => %{"input_tokens" => 10, "output_tokens" => 20}
      }

      req = %Req.Request{
        method: :post,
        url: URI.parse("https://api.openai.com/v1/responses"),
        headers: %{},
        body: {:json, %{}},
        options: %{
          id: "gpt-5",
          operation: :object,
          compiled_schema: compiled_schema
        }
      }

      resp = %Req.Response{status: 200, headers: %{}, body: response_body}

      {_req, decoded_resp} = ResponsesAPI.decode_response({req, resp})

      assert %ReqLLM.Response{} = decoded_resp.body
      assert decoded_resp.body.object == nil
      assert decoded_resp.body.provider_meta[:object_parse_error] == :validation_failed
    end

    test "respects json_repair option for structured output tool calls" do
      schema = [name: [type: :string, required: true]]
      {:ok, compiled_schema} = ReqLLM.Schema.compile(schema)

      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5.4",
        "output" => [
          %{
            "type" => "function_call",
            "call_id" => "call_abc",
            "name" => "structured_output",
            "arguments" => ~s({"name":"Alice",})
          }
        ],
        "usage" => %{"input_tokens" => 10, "output_tokens" => 20}
      }

      req = %Req.Request{
        method: :post,
        url: URI.parse("https://api.openai.com/v1/responses"),
        headers: %{},
        body: {:json, %{}},
        options: %{
          id: "gpt-5",
          operation: :object,
          compiled_schema: compiled_schema,
          json_repair: false
        }
      }

      resp = %Req.Response{status: 200, headers: %{}, body: response_body}

      {_req, decoded_resp} = ResponsesAPI.decode_response({req, resp})

      assert %ReqLLM.Response{} = decoded_resp.body
      assert decoded_resp.body.object == nil
      assert decoded_resp.body.provider_meta[:object_parse_error] == :invalid_json
    end
  end
end
