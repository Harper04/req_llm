defmodule Provider.OpenAI.ResponsesAPI.StreamingTest do
  use ExUnit.Case, async: false

  import ReqLLM.Test.OpenAI.ResponsesAPIHelpers,
    only: [decode_stream_events: 2, phased_message_item: 3, refusal_message_item: 2]

  alias ReqLLM.Providers.OpenAI.ResponsesAPI

  describe "decode_stream_event/2" do
    setup do
      {:ok, model} = ReqLLM.model("openai:gpt-5")
      {:ok, model: model}
    end

    test "decodes output_text delta", %{model: model} do
      event = %{data: %{"event" => "response.output_text.delta", "delta" => "Hello"}}

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.type == :content
      assert chunk.text == "Hello"
    end

    test "ignores empty output_text delta", %{model: model} do
      event = %{data: %{"event" => "response.output_text.delta", "delta" => ""}}

      assert [] = ResponsesAPI.decode_stream_event(event, model)
    end

    test "decodes reasoning delta", %{model: model} do
      event = %{data: %{"event" => "response.reasoning.delta", "delta" => "Thinking..."}}

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.type == :thinking
      assert chunk.text == "Thinking..."
    end

    test "ignores empty reasoning delta", %{model: model} do
      event = %{data: %{"event" => "response.reasoning.delta", "delta" => ""}}

      assert [] = ResponsesAPI.decode_stream_event(event, model)
    end

    test "decodes reasoning summary text delta", %{model: model} do
      event = %{
        data: %{
          "event" => "response.reasoning_summary_text.delta",
          "delta" => "Checking primality..."
        }
      }

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.type == :thinking
      assert chunk.text == "Checking primality..."
    end

    test "ignores empty reasoning summary text delta", %{model: model} do
      event = %{
        data: %{
          "event" => "response.reasoning_summary_text.delta",
          "delta" => ""
        }
      }

      assert [] = ResponsesAPI.decode_stream_event(event, model)
    end

    test "ignores reasoning summary text done event", %{model: model} do
      event = %{data: %{"event" => "response.reasoning_summary_text.done"}}

      assert [] = ResponsesAPI.decode_stream_event(event, model)
    end

    test "reasoning summary deltas carry the item id and part indexes", %{model: model} do
      event = %{
        data: %{
          "event" => "response.reasoning_summary_text.delta",
          "item_id" => "rs_123",
          "output_index" => 0,
          "summary_index" => 1,
          "delta" => "Checking primality..."
        }
      }

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.type == :thinking
      assert chunk.metadata.item_id == "rs_123"
      assert chunk.metadata.output_index == 0
      assert chunk.metadata.summary_index == 1
      assert chunk.metadata.provider_data == %{"type" => "reasoning", "id" => "rs_123"}
      assert chunk.metadata.provider == :openai
      assert chunk.metadata.format == "openai-responses-v1"
    end

    test "decodes reasoning_text delta as thinking", %{model: model} do
      event = %{
        data: %{
          "event" => "response.reasoning_text.delta",
          "item_id" => "rs_123",
          "output_index" => 0,
          "delta" => "Raw reasoning"
        }
      }

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.type == :thinking
      assert chunk.text == "Raw reasoning"
      assert chunk.metadata.item_id == "rs_123"
      refute Map.has_key?(chunk.metadata, :summary_index)

      assert [] =
               ResponsesAPI.decode_stream_event(
                 %{data: %{"event" => "response.reasoning_text.done"}},
                 model
               )
    end

    test "reasoning summary part added marks a part boundary", %{model: model} do
      event = %{
        data: %{
          "event" => "response.reasoning_summary_part.added",
          "item_id" => "rs_123",
          "output_index" => 0,
          "summary_index" => 2,
          "part" => %{"type" => "summary_text", "text" => ""}
        }
      }

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.type == :meta

      assert chunk.metadata.reasoning_summary_part == %{
               status: :added,
               item_id: "rs_123",
               output_index: 0,
               summary_index: 2,
               text: ""
             }
    end

    test "reasoning summary part done carries the completed part text", %{model: model} do
      event = %{
        data: %{
          "event" => "response.reasoning_summary_part.done",
          "item_id" => "rs_123",
          "output_index" => 0,
          "summary_index" => 0,
          "part" => %{"type" => "summary_text", "text" => "Checked the constraints."}
        }
      }

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.type == :meta

      assert chunk.metadata.reasoning_summary_part == %{
               status: :done,
               item_id: "rs_123",
               output_index: 0,
               summary_index: 0,
               text: "Checked the constraints."
             }
    end

    test "summary part meta chunks do not disturb the assembled response", %{model: model} do
      chunks =
        [
          %{
            "event" => "response.reasoning_summary_part.added",
            "item_id" => "rs_1",
            "summary_index" => 0,
            "part" => %{"type" => "summary_text", "text" => ""}
          },
          %{
            "event" => "response.reasoning_summary_text.delta",
            "item_id" => "rs_1",
            "summary_index" => 0,
            "delta" => "Plan"
          },
          %{
            "event" => "response.reasoning_summary_part.done",
            "item_id" => "rs_1",
            "summary_index" => 0,
            "part" => %{"type" => "summary_text", "text" => "Plan"}
          },
          %{"event" => "response.output_text.delta", "delta" => "Answer"},
          %{
            "event" => "response.completed",
            "response" => %{
              "id" => "resp_1",
              "output" => [
                %{
                  "id" => "rs_1",
                  "type" => "reasoning",
                  "summary" => [%{"type" => "summary_text", "text" => "Plan"}]
                },
                %{
                  "type" => "message",
                  "content" => [%{"type" => "output_text", "text" => "Answer"}]
                }
              ],
              "usage" => %{"input_tokens" => 1, "output_tokens" => 2}
            }
          }
        ]
        |> Enum.flat_map(&ResponsesAPI.decode_stream_event(%{data: &1}, model))

      {:ok, response} =
        ReqLLM.Provider.Defaults.ResponseBuilder.build_response(
          chunks,
          %{id: "resp_1", model: "gpt-5"},
          context: ReqLLM.Context.new([]),
          model: model
        )

      assert ReqLLM.Response.text(response) == "Answer"
      assert ReqLLM.Response.thinking(response) == "Plan"
      assert [%ReqLLM.Message.ReasoningDetails{text: "Plan"}] = response.message.reasoning_details
    end

    test "decodes usage event", %{model: model} do
      event = %{
        data: %{
          "event" => "response.usage",
          "usage" => %{
            "input_tokens" => 10,
            "output_tokens" => 20
          }
        }
      }

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.type == :meta
      assert chunk.metadata.usage.input_tokens == 10
      assert chunk.metadata.usage.output_tokens == 20
      assert chunk.metadata.usage.total_tokens == 30
      assert chunk.metadata.model == "gpt-5"
    end

    test "normalizes usage with reasoning_tokens", %{model: model} do
      event = %{
        data: %{
          "event" => "response.usage",
          "usage" => %{
            "input_tokens" => 10,
            "output_tokens" => 20,
            "input_tokens_details" => %{
              "cached_tokens" => 4,
              "cache_write_tokens" => 6
            },
            "output_tokens_details" => %{
              "reasoning_tokens" => 5
            }
          }
        }
      }

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.metadata.usage.input_tokens == 10
      assert chunk.metadata.usage.output_tokens == 20
      assert chunk.metadata.usage.total_tokens == 30
      assert chunk.metadata.usage.cached_tokens == 4
      assert chunk.metadata.usage.cache_creation_tokens == 6
      assert chunk.metadata.usage.reasoning_tokens == 5
    end

    test "ignores output_text done event", %{model: model} do
      event = %{data: %{"event" => "response.output_text.done"}}

      assert [] = ResponsesAPI.decode_stream_event(event, model)
    end

    test "decodes completed function_call output items", %{model: model} do
      event = %{
        data: %{
          "event" => "response.output_item.done",
          "output_index" => 0,
          "item" => %{
            "type" => "function_call",
            "call_id" => "call_123",
            "name" => "get_weather",
            "arguments" => ~s({"location":"SF"})
          }
        }
      }

      assert [tool_chunk, args_chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert tool_chunk.type == :tool_call
      assert tool_chunk.name == "get_weather"
      assert tool_chunk.metadata.id == "call_123"
      assert tool_chunk.metadata.index == 0
      assert args_chunk.metadata.tool_call_args.fragment == ~s({"location":"SF"})
    end

    test "decodes completed message output items", %{model: model} do
      event = %{
        data: %{
          "event" => "response.output_item.done",
          "output_index" => 0,
          "item" => %{
            "type" => "message",
            "content" => [%{"type" => "output_text", "text" => "Final answer"}]
          }
        }
      }

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.type == :content
      assert chunk.text == "Final answer"
    end

    test "marks builtin output item starts for stream server timing", %{model: model} do
      event = %{
        data: %{
          "event" => "response.output_item.added",
          "output_index" => 0,
          "item" => %{
            "id" => "ws_1",
            "type" => "web_search_call"
          }
        }
      }

      assert [%ReqLLM.StreamChunk{type: :meta, metadata: metadata}] =
               ResponsesAPI.decode_stream_event(event, model)

      assert metadata.builtin_tool_started.id == "ws_1"
      assert metadata.builtin_tool_started.name == "web_search_call"
      assert metadata.builtin_tool_started.index == 0
      assert is_integer(metadata.builtin_tool_started.started_at_unix_nano)
    end

    test "marks builtin output item completions for stream server timing", %{model: model} do
      event = %{
        data: %{
          "event" => "response.output_item.done",
          "output_index" => 0,
          "item" => %{
            "id" => "ws_1",
            "type" => "web_search_call",
            "action" => %{"query" => "elixir telemetry"},
            "status" => "completed"
          }
        }
      }

      assert [%ReqLLM.StreamChunk{type: :tool_call} = chunk] =
               ResponsesAPI.decode_stream_event(event, model)

      assert chunk.name == "web_search_call"
      assert chunk.arguments == %{"action" => %{"query" => "elixir telemetry"}}
      assert chunk.metadata.id == "ws_1"
      assert chunk.metadata.builtin? == true
      assert chunk.metadata.status == "completed"
      assert is_integer(chunk.metadata.done_at_unix_nano)
    end

    test "carries a failed builtin item's status on the tool call chunk", %{model: model} do
      event = %{
        data: %{
          "event" => "response.output_item.done",
          "output_index" => 0,
          "item" => %{
            "id" => "ws_1",
            "type" => "web_search_call",
            "action" => %{"query" => "elixir telemetry"},
            "status" => "failed"
          }
        }
      }

      assert [%ReqLLM.StreamChunk{type: :tool_call} = chunk] =
               ResponsesAPI.decode_stream_event(event, model)

      assert chunk.metadata.status == "failed"
      refute Map.has_key?(chunk.arguments, "status")
    end

    test "decodes atom-keyed builtin output item events", %{model: model} do
      added = %{
        event: "response.output_item.added",
        data: %{
          output_index: 0,
          item: %{
            id: "ws_1",
            type: "web_search_call"
          }
        }
      }

      done = %{
        event: "response.output_item.done",
        data: %{
          output_index: 0,
          item: %{
            id: "ws_1",
            type: "web_search_call",
            action: %{"query" => "elixir telemetry"},
            status: "completed"
          }
        }
      }

      assert [%ReqLLM.StreamChunk{type: :meta, metadata: start_meta}] =
               ResponsesAPI.decode_stream_event(added, model)

      assert start_meta.builtin_tool_started.id == "ws_1"

      assert [%ReqLLM.StreamChunk{type: :tool_call} = chunk] =
               ResponsesAPI.decode_stream_event(done, model)

      assert chunk.arguments == %{action: %{"query" => "elixir telemetry"}}
      assert chunk.metadata.id == "ws_1"
      assert chunk.metadata.builtin? == true
      assert is_integer(chunk.metadata.done_at_unix_nano)
    end

    test "emits code_interpreter items from output_item.added", %{model: model} do
      event = %{
        data: %{
          "event" => "response.output_item.added",
          "output_index" => 0,
          "item" => %{
            "id" => "ci_xxx",
            "type" => "code_interpreter_call",
            "code" => "print(2+2)",
            "status" => "in_progress"
          }
        }
      }

      assert [%ReqLLM.StreamChunk{type: :meta, metadata: metadata}] =
               ResponsesAPI.decode_stream_event(event, model)

      assert metadata.code_interpreter_item["type"] == "code_interpreter_call"
      assert metadata.code_interpreter_item["id"] == "ci_xxx"
    end

    test "emits code_interpreter items from output_item.done", %{model: model} do
      event = %{
        data: %{
          "event" => "response.output_item.done",
          "output_index" => 0,
          "item" => %{
            "type" => "code_interpreter_logs",
            "call_id" => "ci_xxx",
            "logs" => "4\n"
          }
        }
      }

      assert [%ReqLLM.StreamChunk{type: :meta, metadata: metadata}] =
               ResponsesAPI.decode_stream_event(event, model)

      assert metadata.code_interpreter_item["type"] == "code_interpreter_logs"
      assert metadata.code_interpreter_item["logs"] == "4\n"
    end

    test "collects code_interpreter items in response.completed provider_meta", %{model: model} do
      completed = %{
        data: %{
          "event" => "response.completed",
          "response" => %{
            "id" => "resp_123",
            "output" => [
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
              }
            ]
          }
        }
      }

      assert [%ReqLLM.StreamChunk{type: :meta, metadata: metadata}] =
               ResponsesAPI.decode_stream_event(completed, model)

      assert metadata.terminal? == true
      assert [call, logs] = get_in(metadata.provider_meta, ["code_interpreter", "items"])
      assert call["type"] == "code_interpreter_call"
      assert logs["type"] == "code_interpreter_logs"
    end

    test "stateful decoding avoids duplicate completed output items", %{model: model} do
      added = %{
        data: %{
          "event" => "response.output_item.added",
          "output_index" => 0,
          "item" => %{
            "type" => "function_call",
            "call_id" => "call_123",
            "name" => "get_weather"
          }
        }
      }

      delta = %{
        data: %{
          "event" => "response.function_call_arguments.delta",
          "output_index" => 0,
          "delta" => ~s({"location":"SF"})
        }
      }

      done = %{
        data: %{
          "event" => "response.output_item.done",
          "output_index" => 0,
          "item" => %{
            "type" => "function_call",
            "call_id" => "call_123",
            "name" => "get_weather",
            "arguments" => ~s({"location":"SF"})
          }
        }
      }

      {added_chunks, state} = ResponsesAPI.decode_stream_event(added, model, nil)
      {delta_chunks, state} = ResponsesAPI.decode_stream_event(delta, model, state)
      {done_chunks, _state} = ResponsesAPI.decode_stream_event(done, model, state)

      assert [%ReqLLM.StreamChunk{type: :tool_call}] = added_chunks
      assert [%ReqLLM.StreamChunk{type: :meta}] = delta_chunks
      assert done_chunks == []
    end

    test "empty text deltas do not suppress completed message output items", %{model: model} do
      delta = %{
        data: %{
          "event" => "response.output_text.delta",
          "output_index" => 0,
          "delta" => ""
        }
      }

      done = %{
        data: %{
          "event" => "response.output_item.done",
          "output_index" => 0,
          "item" => %{
            "type" => "message",
            "content" => [%{"type" => "output_text", "text" => "Final answer"}]
          }
        }
      }

      {delta_chunks, state} = ResponsesAPI.decode_stream_event(delta, model, nil)
      {done_chunks, _state} = ResponsesAPI.decode_stream_event(done, model, state)

      assert delta_chunks == []
      assert [%ReqLLM.StreamChunk{type: :content, text: "Final answer"}] = done_chunks
    end

    test "decodes completed event", %{model: model} do
      event = %{data: %{"event" => "response.completed"}}

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.type == :meta
      assert chunk.metadata.terminal? == true
      assert chunk.metadata.finish_reason == :stop
    end

    test "extracts phase metadata from completed event output items", %{model: model} do
      event = %{
        data: %{
          "event" => "response.completed",
          "response" => %{
            "id" => "resp_123",
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
            ]
          }
        }
      }

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.type == :meta
      assert chunk.metadata.response_id == "resp_123"

      assert chunk.metadata.phase_items == [
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

    test "extracts encrypted reasoning details from completed event output items", %{model: model} do
      event = %{
        data: %{
          "event" => "response.completed",
          "response" => %{
            "id" => "resp_123",
            "output" => [
              %{
                "id" => "rs_123",
                "type" => "reasoning",
                "summary" => [%{"type" => "summary_text", "text" => "Checked constraints."}],
                "encrypted_content" => "encrypted_reasoning_payload"
              },
              %{
                "type" => "message",
                "content" => [%{"type" => "output_text", "text" => "OK"}]
              }
            ]
          }
        }
      }

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.type == :meta
      assert [detail] = chunk.metadata.reasoning_details
      assert detail.text == "Checked constraints."
      assert detail.signature == "encrypted_reasoning_payload"
      assert detail.encrypted? == true
      assert detail.provider == :openai
      assert detail.format == "openai-responses-v1"
      assert detail.index == 0

      assert detail.provider_data == %{
               "id" => "rs_123",
               "type" => "reasoning",
               "summary" => [%{"type" => "summary_text", "text" => "Checked constraints."}]
             }
    end

    test "decodes incomplete event", %{model: model} do
      event = %{data: %{"event" => "response.incomplete", "reason" => "length"}}

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.type == :meta
      assert chunk.metadata.terminal? == true
      assert chunk.metadata.finish_reason == :length
    end

    test "decodes incomplete event with usage so metadata includes usage when finish_reason is length",
         %{model: model} do
      event = %{
        data: %{
          "event" => "response.incomplete",
          "reason" => "length",
          "response" => %{
            "incomplete_details" => %{"reason" => "length"},
            "usage" => %{
              "input_tokens" => 8,
              "output_tokens" => 12
            }
          }
        }
      }

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.type == :meta
      assert chunk.metadata.terminal? == true
      assert chunk.metadata.finish_reason == :length
      assert %{input_tokens: 8, output_tokens: 12, total_tokens: 20} = chunk.metadata.usage
    end

    test "decodes failed event", %{model: model} do
      event = %{
        data: %{
          "event" => "response.failed",
          "response" => %{
            "id" => "resp_123",
            "status" => "failed",
            "error" => %{"code" => "server_error", "message" => "The model run failed"}
          }
        }
      }

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.type == :meta
      assert chunk.metadata.terminal? == true
      assert chunk.metadata.finish_reason == :error
      assert chunk.metadata.error == "The model run failed"
      assert chunk.metadata.error_code == "server_error"
      assert chunk.metadata.response_id == "resp_123"
    end

    test "decodes failed event with usage so token spend is captured", %{model: model} do
      event = %{
        data: %{
          "event" => "response.failed",
          "response" => %{
            "error" => %{"message" => "The model run failed"},
            "usage" => %{"input_tokens" => 8, "output_tokens" => 12}
          }
        }
      }

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.type == :meta
      assert chunk.metadata.terminal? == true
      assert chunk.metadata.finish_reason == :error
      assert %{input_tokens: 8, output_tokens: 12, total_tokens: 20} = chunk.metadata.usage
    end

    test "decodes failed event falling back to error code", %{model: model} do
      event = %{
        data: %{
          "event" => "response.failed",
          "response" => %{"error" => %{"code" => "server_error"}}
        }
      }

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.metadata.terminal? == true
      assert chunk.metadata.finish_reason == :error
      assert chunk.metadata.error == "server_error"
      assert chunk.metadata.error_code == "server_error"
    end

    test "decodes failed event without error details", %{model: model} do
      event = %{data: %{"event" => "response.failed"}}

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.type == :meta
      assert chunk.metadata.terminal? == true
      assert chunk.metadata.finish_reason == :error
      assert chunk.metadata.error == "response failed"
    end

    test "decodes error event", %{model: model} do
      event = %{
        data: %{
          "event" => "error",
          "code" => "rate_limit_exceeded",
          "message" => "Rate limit exceeded",
          "param" => nil
        }
      }

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.type == :meta
      assert chunk.metadata.terminal? == true
      assert chunk.metadata.finish_reason == :error
      assert chunk.metadata.error == "Rate limit exceeded"
    end

    test "decodes error event falling back to code", %{model: model} do
      event = %{data: %{"event" => "error", "code" => "server_error"}}

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.metadata.terminal? == true
      assert chunk.metadata.finish_reason == :error
      assert chunk.metadata.error == "server_error"
    end

    test "decodes error event with nested error details", %{model: model} do
      event = %{
        data: %{
          "event" => "error",
          "type" => "error",
          "error" => %{
            "type" => "invalid_request_error",
            "code" => "context_length_exceeded",
            "message" =>
              "Your input exceeds the context window of this model. Please adjust your input and try again.",
            "param" => "input"
          },
          "sequence_number" => 2
        }
      }

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.metadata.terminal? == true
      assert chunk.metadata.finish_reason == :error
      assert chunk.metadata.error =~ "exceeds the context window"
      assert chunk.metadata.error_code == "context_length_exceeded"
    end

    test "decodes refusal delta as text", %{model: model} do
      event = %{
        data: %{
          "type" => "response.refusal.delta",
          "item_id" => "msg_1",
          "output_index" => 0,
          "content_index" => 0,
          "delta" => "I can't help with that."
        }
      }

      assert [%ReqLLM.StreamChunk{type: :content, text: "I can't help with that."}] =
               ResponsesAPI.decode_stream_event(event, model)

      assert [] =
               ResponsesAPI.decode_stream_event(
                 %{data: %{"type" => "response.refusal.delta", "delta" => ""}},
                 model
               )
    end

    test "refusal done emits nothing", %{model: model} do
      event = %{
        data: %{
          "type" => "response.refusal.done",
          "item_id" => "msg_1",
          "output_index" => 0,
          "content_index" => 0,
          "refusal" => "I can't help with that."
        }
      }

      assert [] = ResponsesAPI.decode_stream_event(event, model)
    end

    test "response.completed with a refusal reports content_filter and the refusal", %{
      model: model
    } do
      event = %{
        data: %{
          "type" => "response.completed",
          "response" => %{
            "id" => "resp_1",
            "status" => "completed",
            "output" => [refusal_message_item("msg_1", "I can't help with that.")]
          }
        }
      }

      assert [%ReqLLM.StreamChunk{type: :meta, metadata: metadata}] =
               ResponsesAPI.decode_stream_event(event, model)

      assert %{terminal?: true, finish_reason: :content_filter} = metadata
      assert metadata.provider_meta["refusals"] == ["I can't help with that."]
    end

    test "response.completed without a refusal still reports stop", %{model: model} do
      event = %{
        data: %{
          "type" => "response.completed",
          "response" => %{
            "id" => "resp_1",
            "status" => "completed",
            "output" => [phased_message_item("msg_1", nil, "Hello")]
          }
        }
      }

      assert [%ReqLLM.StreamChunk{type: :meta, metadata: metadata}] =
               ResponsesAPI.decode_stream_event(event, model)

      assert metadata.finish_reason == :stop
      refute Map.has_key?(metadata[:provider_meta] || %{}, "refusals")
    end

    test "response.incomplete with a refusal keeps its finish reason", %{model: model} do
      event = %{
        data: %{
          "type" => "response.incomplete",
          "response" => %{
            "id" => "resp_1",
            "status" => "incomplete",
            "incomplete_details" => %{"reason" => "max_output_tokens"},
            "output" => [refusal_message_item("msg_1", "I can't help")]
          }
        }
      }

      assert [%ReqLLM.StreamChunk{type: :meta, metadata: metadata}] =
               ResponsesAPI.decode_stream_event(event, model)

      assert metadata.finish_reason == :length
      assert metadata.provider_meta["refusals"] == ["I can't help"]
    end

    test "stateful decoding stamps the message phase on refusal deltas", %{model: model} do
      item =
        "msg_1"
        |> refusal_message_item("I can't help with that.")
        |> Map.put("phase", "final_answer")

      events = [
        %{
          "type" => "response.output_item.added",
          "output_index" => 0,
          "item" => Map.merge(item, %{"status" => "in_progress", "content" => []})
        },
        %{
          "type" => "response.refusal.delta",
          "item_id" => "msg_1",
          "output_index" => 0,
          "content_index" => 0,
          "delta" => "I can't help with that."
        },
        %{"type" => "response.output_item.done", "output_index" => 0, "item" => item}
      ]

      assert [
               %ReqLLM.StreamChunk{
                 type: :content,
                 text: "I can't help with that.",
                 metadata: %{phase: "final_answer", output_index: 0}
               }
             ] = decode_stream_events(events, model)
    end

    test "stateful decoding streams a refusal once and ends with content_filter", %{
      model: model
    } do
      item = refusal_message_item("msg_1", "I'm sorry, I can't help with that.")

      events = [
        %{
          "type" => "response.output_item.added",
          "output_index" => 0,
          "item" => Map.merge(item, %{"status" => "in_progress", "content" => []})
        },
        %{
          "type" => "response.refusal.delta",
          "item_id" => "msg_1",
          "output_index" => 0,
          "content_index" => 0,
          "delta" => "I'm sorry, "
        },
        %{
          "type" => "response.refusal.delta",
          "item_id" => "msg_1",
          "output_index" => 0,
          "content_index" => 0,
          "delta" => "I can't help with that."
        },
        %{
          "type" => "response.refusal.done",
          "item_id" => "msg_1",
          "output_index" => 0,
          "content_index" => 0,
          "refusal" => "I'm sorry, I can't help with that."
        },
        %{"type" => "response.output_item.done", "output_index" => 0, "item" => item},
        %{
          "type" => "response.completed",
          "response" => %{"id" => "resp_1", "status" => "completed", "output" => [item]}
        }
      ]

      assert [
               %ReqLLM.StreamChunk{type: :content, text: "I'm sorry, "},
               %ReqLLM.StreamChunk{type: :content, text: "I can't help with that."},
               %ReqLLM.StreamChunk{
                 type: :meta,
                 metadata: %{terminal?: true, finish_reason: :content_filter}
               }
             ] = decode_stream_events(events, model)
    end

    test "stateful decoding emits refusal text from the done item when no deltas arrived", %{
      model: model
    } do
      item = refusal_message_item("msg_1", "I can't help with that.")

      events = [
        %{"type" => "response.output_item.done", "output_index" => 0, "item" => item}
      ]

      assert [%ReqLLM.StreamChunk{type: :content, text: "I can't help with that."}] =
               decode_stream_events(events, model)
    end

    test "decodes error event without details", %{model: model} do
      event = %{data: %{"event" => "error"}}

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.metadata.terminal? == true
      assert chunk.metadata.finish_reason == :error
      assert chunk.metadata.error == "stream error"
    end

    test "handles [DONE] event", %{model: model} do
      event = %{data: "[DONE]"}

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.type == :meta
      assert chunk.metadata.terminal? == true
    end

    test "uses type field when event field missing", %{model: model} do
      event = %{data: %{"type" => "response.output_text.delta", "delta" => "Text"}}

      assert [chunk] = ResponsesAPI.decode_stream_event(event, model)
      assert chunk.type == :content
      assert chunk.text == "Text"
    end

    test "ignores unknown event types", %{model: model} do
      event = %{data: %{"event" => "response.unknown.type"}}

      assert [] = ResponsesAPI.decode_stream_event(event, model)
    end

    test "ignores events with missing event type", %{model: model} do
      event = %{data: %{"delta" => "text"}}

      assert [] = ResponsesAPI.decode_stream_event(event, model)
    end
  end

  describe "attach_websocket_stream/3" do
    test "builds websocket create event from standard responses request body" do
      original_key = System.get_env("OPENAI_API_KEY")
      System.put_env("OPENAI_API_KEY", "test-key-12345")

      on_exit(fn ->
        if original_key do
          System.put_env("OPENAI_API_KEY", original_key)
        else
          System.delete_env("OPENAI_API_KEY")
        end
      end)

      {:ok, model} = ReqLLM.model("openai:gpt-5")
      context = ReqLLM.Context.new([ReqLLM.Context.user("Say hello")])

      {:ok, config} =
        ResponsesAPI.attach_websocket_stream(
          model,
          context,
          base_url: "http://localhost:4010/v1",
          provider_options: []
        )

      assert config.url == "ws://localhost:4010/v1/responses"
      assert [{"Authorization", "Bearer test-key-12345"}] = config.headers
      assert %ReqLLM.Streaming.Fixtures.HTTPContext{} = config.http_context
      assert config.canonical_json["model"] == "gpt-5"
      refute Map.has_key?(config.canonical_json, "stream")

      [message] = config.initial_messages
      payload = Jason.decode!(message)

      assert payload["type"] == "response.create"
      assert Map.delete(payload, "type") == config.canonical_json
      refute Map.has_key?(payload, "response")
    end

    test "keeps previous_response_id with store false over websocket transport" do
      original_key = System.get_env("OPENAI_API_KEY")
      System.put_env("OPENAI_API_KEY", "test-key-12345")

      on_exit(fn ->
        if original_key do
          System.put_env("OPENAI_API_KEY", original_key)
        else
          System.delete_env("OPENAI_API_KEY")
        end
      end)

      {:ok, model} = ReqLLM.model("openai:gpt-5")

      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Previous answer"}],
        metadata: %{response_id: "resp_ws_123"}
      }

      user_msg = %ReqLLM.Message{
        role: :user,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Follow up"}]
      }

      context = %ReqLLM.Context{messages: [assistant_msg, user_msg]}

      {:ok, config} =
        ResponsesAPI.attach_websocket_stream(
          model,
          context,
          base_url: "http://localhost:4010/v1",
          provider_options: [store: false]
        )

      assert config.canonical_json["previous_response_id"] == "resp_ws_123"
      assert config.canonical_json["store"] == false
    end
  end
end
