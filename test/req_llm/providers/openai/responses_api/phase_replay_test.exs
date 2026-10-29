defmodule Provider.OpenAI.ResponsesAPI.PhaseReplayTest do
  use ExUnit.Case, async: false

  import ReqLLM.Test.OpenAI.ResponsesAPIHelpers,
    only: [build_request: 1, build_response: 2, decode_stream_events: 2, phased_message_item: 3]

  alias ReqLLM.Providers.OpenAI.ResponsesAPI

  describe "phase replay - encode_body/1" do
    test "includes phase when assistant metadata has a single phase" do
      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Root cause found"}],
        metadata: %{phase: "final_answer", response_id: "resp_prev_123"}
      }

      user_msg = %ReqLLM.Message{
        role: :user,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Continue"}]
      }

      context = %ReqLLM.Context{messages: [assistant_msg, user_msg]}
      request = build_request(context: context, provider_options: [store: false])

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert Enum.at(body["input"], 0) == %{
               "role" => "assistant",
               "phase" => "final_answer",
               "content" => [%{"type" => "output_text", "text" => "Root cause found"}]
             }
    end

    test "replays ordered phase_items from assistant metadata" do
      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [
          %ReqLLM.Message.ContentPart{
            type: :text,
            text: "Inspecting logs. Cache race confirmed."
          }
        ],
        metadata: %{
          phase_items: [
            %{
              "phase" => "commentary",
              "content" => [%{"type" => "output_text", "text" => "Inspecting logs. "}]
            },
            %{
              "phase" => "final_answer",
              "content" => [%{"type" => "output_text", "text" => "Cache race confirmed."}]
            }
          ]
        }
      }

      user_msg = %ReqLLM.Message{
        role: :user,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Continue"}]
      }

      context = %ReqLLM.Context{messages: [assistant_msg, user_msg]}
      request = build_request(context: context, provider_options: [store: false])

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert Enum.take(body["input"], 2) == [
               %{
                 "role" => "assistant",
                 "phase" => "commentary",
                 "content" => [%{"type" => "output_text", "text" => "Inspecting logs. "}]
               },
               %{
                 "role" => "assistant",
                 "phase" => "final_answer",
                 "content" => [%{"type" => "output_text", "text" => "Cache race confirmed."}]
               }
             ]
    end

    test "round-trips phased output through decode and encode" do
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

      {_req, decoded_resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      request =
        build_request(context: decoded_resp.body.context, provider_options: [store: false])

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert body["input"] == [
               %{
                 "role" => "assistant",
                 "phase" => "commentary",
                 "content" => [%{"type" => "output_text", "text" => "Inspecting logs. "}]
               },
               %{
                 "role" => "assistant",
                 "phase" => "final_answer",
                 "content" => [%{"type" => "output_text", "text" => "Cache race confirmed."}]
               }
             ]
    end
  end

  describe "assistant phase on content parts" do
    alias ReqLLM.Message.ContentPart
    alias ReqLLM.Providers.OpenAI.ResponsesAPI.ResponseBuilder

    @preamble "Progress 2/5: all checks point to the same issue, so I’m consolidating the diagnosis."
    @answer "It failed because the image can’t be pulled.\n\nRoot cause: `ImagePullBackOff`"

    setup do
      {:ok, model} = ReqLLM.model("openai:gpt-5")
      {:ok, model: model}
    end

    test "decodes each phased message item into its own text part" do
      response_body = %{
        "id" => "resp_phase_1",
        "model" => "gpt-5.4",
        "output" => [
          phased_message_item("msg_1", "commentary", @preamble),
          phased_message_item("msg_2", "final_answer", @answer)
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert resp.body.message.content == [
               ContentPart.text(@preamble, %{phase: "commentary"}),
               ContentPart.text(@answer, %{phase: "final_answer"})
             ]

      assert ReqLLM.Response.text(resp.body) == @preamble <> @answer

      assert resp.body.message.metadata[:phase_items] == [
               %{
                 "phase" => "commentary",
                 "content" => [%{"type" => "output_text", "text" => @preamble}]
               },
               %{
                 "phase" => "final_answer",
                 "content" => [%{"type" => "output_text", "text" => @answer}]
               }
             ]
    end

    test "keeps a single aggregated text part when no item carries a phase" do
      response_body = %{
        "id" => "resp_phase_2",
        "model" => "gpt-5.4",
        "output" => [
          %{
            "type" => "message",
            "content" => [%{"type" => "output_text", "text" => "First item. "}]
          },
          %{
            "type" => "message",
            "content" => [%{"type" => "output_text", "text" => "Second item."}]
          }
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert resp.body.message.content == [
               %ContentPart{type: :text, text: "First item. Second item."}
             ]
    end

    test "keeps interleaved commentary items separate and in order" do
      response_body = %{
        "id" => "resp_phase_3",
        "model" => "gpt-5.6-terra",
        "output" => [
          phased_message_item(
            "msg_1",
            "commentary",
            "I’ll investigate the failed deployment broadly."
          ),
          %{"id" => "rs_1", "type" => "reasoning", "summary" => [], "content" => []},
          phased_message_item(
            "msg_2",
            "commentary",
            "**Progress 1/5 — gathering the deployment evidence in parallel.**"
          ),
          %{
            "id" => "fc_1",
            "type" => "function_call",
            "call_id" => "call_1",
            "name" => "inspect_resource",
            "arguments" => ~s({"resource":"deployment"}),
            "status" => "completed"
          },
          %{
            "id" => "fc_2",
            "type" => "function_call",
            "call_id" => "call_2",
            "name" => "inspect_resource",
            "arguments" => ~s({"resource":"pod_logs"}),
            "status" => "completed"
          }
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert resp.body.message.content == [
               ContentPart.text(
                 "I’ll investigate the failed deployment broadly.",
                 %{phase: "commentary"}
               ),
               ContentPart.text(
                 "**Progress 1/5 — gathering the deployment evidence in parallel.**",
                 %{phase: "commentary"}
               )
             ]

      assert [%{id: "call_1"}, %{id: "call_2"}] = resp.body.message.tool_calls
    end

    test "labels only the items that carry a phase in a mixed response" do
      response_body = %{
        "id" => "resp_phase_4",
        "model" => "gpt-5.4",
        "output" => [
          phased_message_item("msg_1", "commentary", @preamble),
          %{
            "type" => "message",
            "content" => [%{"type" => "output_text", "text" => @answer}]
          }
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert resp.body.message.content == [
               ContentPart.text(@preamble, %{phase: "commentary"}),
               ContentPart.text(@answer)
             ]

      refute Map.has_key?(resp.body.message.metadata, :phase)
      refute Map.has_key?(resp.body.message.metadata, :phase_items)
    end

    test "stamps streamed text with the phase and output index of its item", %{model: model} do
      events =
        phased_stream_events(0, "msg_1", "commentary", ["Progress 2/5: ", "consolidating."]) ++
          phased_stream_events(1, "msg_2", "final_answer", ["It failed ", "to pull."])

      chunks = decode_stream_events(events, model)
      content_chunks = Enum.filter(chunks, &(&1.type == :content))

      assert Enum.map(content_chunks, &{&1.text, &1.metadata}) == [
               {"Progress 2/5: ", %{phase: "commentary", output_index: 0}},
               {"consolidating.", %{phase: "commentary", output_index: 0}},
               {"It failed ", %{phase: "final_answer", output_index: 1}},
               {"to pull.", %{phase: "final_answer", output_index: 1}}
             ]

      {:ok, response} =
        ResponseBuilder.build_response(chunks, %{finish_reason: :stop},
          context: %ReqLLM.Context{messages: []},
          model: model
        )

      assert response.message.content == [
               ContentPart.text("Progress 2/5: consolidating.", %{phase: "commentary"}),
               ContentPart.text("It failed to pull.", %{phase: "final_answer"})
             ]
    end

    test "keeps consecutive streamed commentary items as separate parts", %{model: model} do
      events =
        phased_stream_events(0, "msg_1", "commentary", ["Checking logs."]) ++
          phased_stream_events(1, "msg_2", "commentary", ["Checking events."])

      chunks = decode_stream_events(events, model)

      {:ok, response} =
        ResponseBuilder.build_response(chunks, %{finish_reason: :stop},
          context: %ReqLLM.Context{messages: []},
          model: model
        )

      assert response.message.content == [
               ContentPart.text("Checking logs.", %{phase: "commentary"}),
               ContentPart.text("Checking events.", %{phase: "commentary"})
             ]
    end

    test "leaves unphased streamed text unstamped and merged", %{model: model} do
      events =
        phased_stream_events(0, "msg_1", nil, ["Hello, "]) ++
          phased_stream_events(1, "msg_2", nil, ["world."])

      chunks = decode_stream_events(events, model)

      assert chunks
             |> Enum.filter(&(&1.type == :content))
             |> Enum.all?(&(&1.metadata == %{}))

      {:ok, response} =
        ResponseBuilder.build_response(chunks, %{finish_reason: :stop},
          context: %ReqLLM.Context{messages: []},
          model: model
        )

      assert response.message.content == [%ContentPart{type: :text, text: "Hello, world."}]
    end

    test "stamps a message item that arrives only on output_item.done", %{model: model} do
      done = %{
        data: %{
          "type" => "response.output_item.done",
          "output_index" => 2,
          "item" => phased_message_item("msg_1", "final_answer", @answer)
        }
      }

      {chunks, _state} = ResponsesAPI.decode_stream_event(done, model, nil)

      assert [%ReqLLM.StreamChunk{type: :content, text: @answer, metadata: metadata}] = chunks
      assert metadata == %{phase: "final_answer", output_index: 2}
    end

    test "encodes consecutive text parts with the same phase as one item" do
      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [
          ContentPart.text("Checking logs. ", %{phase: "commentary"}),
          ContentPart.thinking("Weighing causes."),
          ContentPart.text("Checking events.", %{phase: "commentary"}),
          ContentPart.text(@answer, %{phase: "final_answer"})
        ]
      }

      assert encode_input([assistant_msg]) == [
               %{
                 "role" => "assistant",
                 "phase" => "commentary",
                 "content" => [
                   %{"type" => "output_text", "text" => "Checking logs. "},
                   %{"type" => "output_text", "text" => "Checking events."}
                 ]
               },
               %{
                 "role" => "assistant",
                 "phase" => "final_answer",
                 "content" => [%{"type" => "output_text", "text" => @answer}]
               }
             ]
    end

    test "encodes unphased parts beside phased ones as items without a phase" do
      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [
          ContentPart.text(@preamble, %{phase: "commentary"}),
          ContentPart.text(@answer)
        ]
      }

      assert encode_input([assistant_msg]) == [
               %{
                 "role" => "assistant",
                 "phase" => "commentary",
                 "content" => [%{"type" => "output_text", "text" => @preamble}]
               },
               %{
                 "role" => "assistant",
                 "content" => [%{"type" => "output_text", "text" => @answer}]
               }
             ]
    end

    test "encodes a message without phases as a single item" do
      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [ContentPart.text("First item. "), ContentPart.text("Second item.")]
      }

      assert encode_input([assistant_msg]) == [
               %{
                 "role" => "assistant",
                 "content" => [
                   %{"type" => "output_text", "text" => "First item. "},
                   %{"type" => "output_text", "text" => "Second item."}
                 ]
               }
             ]
    end

    test "round-trips phased items through content parts alone" do
      output = [
        phased_message_item("msg_1", "commentary", @preamble),
        phased_message_item("msg_2", "final_answer", @answer),
        phased_message_item("msg_3", "commentary", "Progress 3/5: verifying.")
      ]

      response_body = %{
        "id" => "resp_phase_5",
        "model" => "gpt-5.4",
        "output" => output,
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))
      stripped = %{resp.body.message | metadata: %{}}

      assert encode_input([stripped]) ==
               Enum.map(output, fn item ->
                 %{
                   "role" => "assistant",
                   "phase" => item["phase"],
                   "content" => [
                     %{"type" => "output_text", "text" => hd(item["content"])["text"]}
                   ]
                 }
               end)
    end

    test "does not leak the phase into an Anthropic request" do
      {:ok, model} = ReqLLM.model("anthropic:claude-sonnet-4-5-20250929")

      context =
        ReqLLM.Context.new([
          ReqLLM.Context.user("Why did it fail?"),
          %ReqLLM.Message{
            role: :assistant,
            content: [
              ContentPart.text(@preamble, %{phase: "commentary"}),
              ContentPart.text(@answer, %{phase: "final_answer"})
            ]
          },
          ReqLLM.Context.user("Continue")
        ])

      request = ReqLLM.Providers.Anthropic.Context.encode_request(context, model)
      assistant = Enum.find(request[:messages], &(&1[:role] == "assistant"))

      assert assistant[:content] == [
               %{type: "text", text: @preamble},
               %{type: "text", text: @answer}
             ]
    end
  end

  defp phased_stream_events(output_index, id, phase, deltas) do
    item = phased_message_item(id, phase, Enum.join(deltas))
    added_item = Map.merge(item, %{"status" => "in_progress", "content" => []})

    [
      %{
        "type" => "response.output_item.added",
        "output_index" => output_index,
        "item" => added_item
      }
    ] ++
      Enum.map(deltas, fn delta ->
        %{
          "type" => "response.output_text.delta",
          "item_id" => id,
          "output_index" => output_index,
          "content_index" => 0,
          "delta" => delta
        }
      end) ++
      [%{"type" => "response.output_item.done", "output_index" => output_index, "item" => item}]
  end

  defp encode_input(messages) do
    request =
      build_request(
        context: %ReqLLM.Context{messages: messages},
        provider_options: [store: false]
      )

    request
    |> ResponsesAPI.encode_body()
    |> ReqLLM.Test.Helpers.json_body()
    |> Map.fetch!("input")
  end
end
