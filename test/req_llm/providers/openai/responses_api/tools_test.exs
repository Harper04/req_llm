defmodule Provider.OpenAI.ResponsesAPI.ToolsTest do
  use ExUnit.Case, async: false

  import ReqLLM.Test.OpenAI.ResponsesAPIHelpers, only: [build_request: 1, build_response: 2]

  alias ReqLLM.Providers.OpenAI.ResponsesAPI

  describe "encode_body/1" do
    test "encodes tools when present" do
      tool =
        ReqLLM.Tool.new!(
          name: "get_weather",
          description: "Get weather",
          parameter_schema: [
            location: [type: :string, required: true]
          ],
          callback: fn _ -> {:ok, "result"} end
        )

      request = build_request(tools: [tool])

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert [encoded_tool] = body["tools"]
      assert encoded_tool["type"] == "function"
      assert encoded_tool["name"] == "get_weather"
      assert encoded_tool["description"] == "Get weather"
      assert encoded_tool["parameters"]["properties"]["location"]["type"] == "string"
    end

    test "preserves required and optional fields on non-strict tools" do
      tool =
        ReqLLM.Tool.new!(
          name: "get_weather",
          description: "Get weather",
          parameter_schema: [
            location: [type: :string, required: true],
            units: [type: :string]
          ],
          callback: fn _ -> {:ok, "result"} end,
          strict: false
        )

      request = build_request(tools: [tool])

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert [encoded_tool] = body["tools"]
      assert encoded_tool["strict"] == false
      assert encoded_tool["parameters"]["required"] == ["location"]
      assert encoded_tool["parameters"]["additionalProperties"] == false
      assert encoded_tool["parameters"]["properties"]["units"] == %{"type" => "string"}
    end

    test "preserves non-strict JSON schemas with definitions and root constraints" do
      parameters = %{
        "$schema" => "https://json-schema.org/draft/2020-12/schema",
        "$defs" => %{"location" => %{"type" => "string", "minLength" => 1}},
        "type" => "object",
        "properties" => %{
          "location" => %{"$ref" => "#/$defs/location"},
          "units" => %{"type" => "string"}
        },
        "required" => ["location"],
        "minProperties" => 1
      }

      tool =
        ReqLLM.Tool.new!(
          name: "get_weather",
          description: "Get weather",
          parameter_schema: parameters,
          callback: fn _ -> {:ok, "result"} end
        )

      body =
        build_request(tools: [tool])
        |> ResponsesAPI.encode_body()
        |> ReqLLM.Test.Helpers.json_body()

      assert [encoded_tool] = body["tools"]
      assert encoded_tool["strict"] == false
      assert encoded_tool["parameters"] == parameters
    end

    test "preserves explicit additionalProperties values on non-strict tools" do
      for additional_properties <- [false, true, %{"type" => "string"}] do
        parameters = %{
          "type" => "object",
          "properties" => %{"location" => %{"type" => "string"}},
          "additionalProperties" => additional_properties
        }

        tool =
          ReqLLM.Tool.new!(
            name: "get_weather",
            description: "Get weather",
            parameter_schema: parameters,
            callback: fn _ -> {:ok, "result"} end
          )

        body =
          build_request(tools: [tool])
          |> ResponsesAPI.encode_body()
          |> ReqLLM.Test.Helpers.json_body()

        assert [encoded_tool] = body["tools"]
        assert encoded_tool["parameters"] == parameters
      end
    end

    test "preserves atom-keyed non-strict schemas and nested constraints" do
      parameters = %{
        type: "object",
        properties: %{
          location: %{
            anyOf: [
              %{type: "string", minLength: 1},
              %{
                type: "object",
                properties: %{city: %{type: "string"}},
                required: ["city"],
                additionalProperties: true
              }
            ]
          }
        },
        required: ["location"],
        additionalProperties: false
      }

      tool =
        ReqLLM.Tool.new!(
          name: "get_weather",
          description: "Get weather",
          parameter_schema: parameters,
          callback: fn _ -> {:ok, "result"} end
        )

      body =
        build_request(tools: [tool])
        |> ResponsesAPI.encode_body()
        |> ReqLLM.Test.Helpers.json_body()

      assert [encoded_tool] = body["tools"]
      assert encoded_tool["parameters"] == parameters |> Jason.encode!() |> Jason.decode!()
    end

    test "does not silently remove malformed non-strict schema fields" do
      parameters = %{
        "type" => "object",
        "properties" => %{"location" => %{"type" => "string"}},
        "required" => "location"
      }

      tool =
        ReqLLM.Tool.new!(
          name: "get_weather",
          description: "Get weather",
          parameter_schema: parameters,
          callback: fn _ -> {:ok, "result"} end
        )

      body =
        build_request(tools: [tool])
        |> ResponsesAPI.encode_body()
        |> ReqLLM.Test.Helpers.json_body()

      assert [encoded_tool] = body["tools"]
      assert encoded_tool["parameters"] == parameters
    end

    test "keeps the closed empty schema for parameterless tools" do
      for strict <- [false, true] do
        tool =
          ReqLLM.Tool.new!(
            name: "get_time",
            description: "Get time",
            strict: strict,
            callback: fn _ -> {:ok, "result"} end
          )

        body =
          build_request(tools: [tool])
          |> ResponsesAPI.encode_body()
          |> ReqLLM.Test.Helpers.json_body()

        assert [encoded_tool] = body["tools"]
        assert encoded_tool["strict"] == strict
        assert encoded_tool["parameters"]["type"] == "object"
        assert encoded_tool["parameters"]["properties"] == %{}
        assert encoded_tool["parameters"]["additionalProperties"] == false
        assert Map.has_key?(encoded_tool["parameters"], "required") == strict
      end
    end

    test "respects explicit false on flat and nested raw tool maps" do
      parameters = %{
        "type" => "object",
        "properties" => %{
          "location" => %{"type" => "string"},
          "units" => %{"type" => "string"}
        },
        "required" => ["location"],
        "additionalProperties" => true
      }

      function = %{
        "name" => "get_weather",
        "description" => "Get weather",
        "parameters" => parameters
      }

      for tool <- [
            Map.merge(function, %{"type" => "function", "strict" => false}),
            %{"type" => "function", "function" => Map.put(function, "strict", false)},
            %{"type" => "function", "strict" => false, "function" => function},
            %{
              "type" => "function",
              "strict" => true,
              "function" => Map.put(function, "strict", false)
            },
            %{
              type: :function,
              strict: false,
              function: %{name: "get_weather", parameters: parameters}
            }
          ] do
        body =
          build_request(tools: [tool])
          |> ResponsesAPI.encode_body()
          |> ReqLLM.Test.Helpers.json_body()

        assert [encoded_tool] = body["tools"]
        assert encoded_tool["strict"] == false
        assert encoded_tool["parameters"] == parameters
      end
    end

    test "keeps raw tool maps strict by default and honors nested explicit true" do
      function = %{
        "name" => "get_weather",
        "parameters" => %{
          "type" => "object",
          "properties" => %{
            "location" => %{"type" => "string"},
            "units" => %{"type" => "string"}
          },
          "required" => ["location"],
          "additionalProperties" => true
        }
      }

      for tool <- [
            Map.put(function, "type", "function"),
            Map.merge(function, %{"type" => "function", "strict" => true}),
            %{"type" => "function", "function" => function},
            %{
              "type" => "function",
              "strict" => false,
              "function" => Map.put(function, "strict", true)
            }
          ] do
        body =
          build_request(tools: [tool])
          |> ResponsesAPI.encode_body()
          |> ReqLLM.Test.Helpers.json_body()

        assert [encoded_tool] = body["tools"]
        assert encoded_tool["strict"] == true
        assert Enum.sort(encoded_tool["parameters"]["required"]) == ["location", "units"]
        assert encoded_tool["parameters"]["additionalProperties"] == false
      end
    end

    test "retains defaults for raw tools without parameters and preserves explicit empty schemas" do
      for strict <- [false, true] do
        tool = %{"type" => "function", "name" => "get_time", "strict" => strict}

        body =
          build_request(tools: [tool])
          |> ResponsesAPI.encode_body()
          |> ReqLLM.Test.Helpers.json_body()

        assert [encoded_tool] = body["tools"]
        assert encoded_tool["parameters"]["type"] == "object"
        assert encoded_tool["parameters"]["properties"] == %{}
        assert encoded_tool["parameters"]["additionalProperties"] == false
        assert Map.has_key?(encoded_tool["parameters"], "required") == strict
      end

      tool = %{"type" => "function", "name" => "get_time", "strict" => false, "parameters" => %{}}

      body =
        build_request(tools: [tool])
        |> ResponsesAPI.encode_body()
        |> ReqLLM.Test.Helpers.json_body()

      assert [%{"parameters" => %{}}] = body["tools"]
      assert hd(body["tools"])["parameters"] == %{}
    end

    test "encodes strict tools with required field listing all parameters" do
      tool =
        ReqLLM.Tool.new!(
          name: "get_weather",
          description: "Get weather",
          parameter_schema: [
            location: [type: :string, required: true],
            units: [type: :string]
          ],
          callback: fn _ -> {:ok, "result"} end,
          strict: true
        )

      request = build_request(tools: [tool])

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert [encoded_tool] = body["tools"]
      assert encoded_tool["strict"] == true
      assert Enum.sort(encoded_tool["parameters"]["required"]) == ["location", "units"]
    end

    test "strictifies raw function tools with JSON schema placeholder parameters" do
      tool = %{
        "type" => "function",
        "function" => %{
          "name" => "websearch",
          "description" => "Search the web",
          "parameters" => %{
            "type" => "object",
            "required" => ["query"],
            "properties" => %{
              "query" => %{"type" => "string"},
              "outputSchema" => %{
                "description" => "JSON schema for synthesized output.content"
              },
              "summary" => %{
                "anyOf" => [
                  %{"type" => "boolean"},
                  %{
                    "type" => "object",
                    "properties" => %{
                      "schema" => %{
                        "description" => "JSON schema for structured summary output"
                      }
                    }
                  }
                ]
              }
            }
          }
        }
      }

      request = build_request(tools: [tool])

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert [encoded_tool] = body["tools"]
      assert encoded_tool["type"] == "function"
      assert encoded_tool["name"] == "websearch"
      assert encoded_tool["strict"] == true

      params = encoded_tool["parameters"]
      assert Enum.sort(params["required"]) == ["outputSchema", "query", "summary"]
      assert params["additionalProperties"] == false

      output_schema = params["properties"]["outputSchema"]
      assert output_schema["description"] == "JSON schema for synthesized output.content"
      assert output_schema["type"] == "object"
      assert output_schema["properties"] == %{}
      assert output_schema["required"] == []
      assert output_schema["additionalProperties"] == false

      [_, summary_object] = params["properties"]["summary"]["anyOf"]
      nested_schema = summary_object["properties"]["schema"]
      assert nested_schema["description"] == "JSON schema for structured summary output"
      assert nested_schema["type"] == "object"
      assert nested_schema["properties"] == %{}
      assert nested_schema["required"] == []
      assert nested_schema["additionalProperties"] == false
    end

    test "passes through code_interpreter tool maps unchanged" do
      tool = %{
        "type" => "code_interpreter",
        "container" => %{"type" => "auto", "memory_limit" => "4g"}
      }

      request = build_request(tools: [tool])

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert [encoded_tool] = body["tools"]
      assert encoded_tool["type"] == "code_interpreter"
      assert encoded_tool["container"] == %{"type" => "auto", "memory_limit" => "4g"}
      refute Map.has_key?(encoded_tool, "strict")
      refute Map.has_key?(encoded_tool, "parameters")
    end

    test "passes through atom-keyed code_interpreter tool maps" do
      tool = %{type: :code_interpreter, container: %{type: :auto, memory_limit: "4g"}}

      request = build_request(tools: [tool])

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert [encoded_tool] = body["tools"]
      assert encoded_tool["type"] == "code_interpreter"
      assert encoded_tool["container"] == %{"type" => "auto", "memory_limit" => "4g"}
    end

    test "passes through string container IDs for code_interpreter" do
      tool = %{"type" => "code_interpreter", "container" => "cntr_abc123"}

      request = build_request(tools: [tool])

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert [encoded_tool] = body["tools"]
      assert encoded_tool["type"] == "code_interpreter"
      assert encoded_tool["container"] == "cntr_abc123"
    end

    test "omits tools when empty list" do
      request = build_request(tools: [])

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      refute Map.has_key?(body, "tools")
    end

    test "encodes tool_choice auto" do
      request = build_request(tool_choice: :auto)

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert body["tool_choice"] == "auto"
    end

    test "encodes tool_choice none" do
      request = build_request(tool_choice: :none)

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert body["tool_choice"] == "none"
    end

    test "encodes tool_choice required" do
      request = build_request(tool_choice: :required)

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert body["tool_choice"] == "required"
    end

    test "encodes structured tool outputs from context metadata" do
      tool_call = %ReqLLM.ToolCall{
        id: "call_1",
        type: "function",
        function: %{name: "get_weather", arguments: ~s({"location":"SF"})}
      }

      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [],
        tool_calls: [tool_call]
      }

      tool_result =
        ReqLLM.Context.tool_result_message(
          "get_weather",
          "call_1",
          %ReqLLM.ToolResult{output: %{temp: 72}}
        )

      context = %ReqLLM.Context{messages: [assistant_msg, tool_result]}
      request = build_request(context: context)

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      tool_output =
        Enum.find(body["input"], fn item ->
          item["type"] == "function_call_output"
        end)

      assert tool_output["call_id"] == "call_1"
      assert Jason.decode!(tool_output["output"]) == %{"temp" => 72}
    end

    test "prefers explicit model-facing content over application output" do
      tool_result =
        ReqLLM.Context.tool_result(
          "call_1",
          "search_documents",
          %ReqLLM.ToolResult{
            output: %{records: [%{id: 1}], internal_cursor: "cursor_123"},
            content: [ReqLLM.Message.ContentPart.text("One matching document was found.")]
          }
        )

      context = %ReqLLM.Context{messages: [tool_result]}
      request = build_request(context: context)

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      tool_output =
        Enum.find(body["input"], fn item ->
          item["type"] == "function_call_output"
        end)

      assert tool_output["output"] == "One matching document was found."
    end

    test "skips builtin tool calls when replaying assistant context" do
      contexts = [
        %ReqLLM.Context{
          messages: [
            %ReqLLM.Message{
              role: :assistant,
              content: [],
              tool_calls: [
                ReqLLM.ToolCall.new_builtin("ws_1", "web_search_call", ~s({"query":"elixir"}))
              ]
            }
          ]
        },
        %ReqLLM.Context{
          messages: [
            %ReqLLM.Message{
              role: :assistant,
              content: [],
              tool_calls: [
                %{
                  "id" => "ws_1",
                  "function" => %{
                    "name" => "web_search_call",
                    "arguments" => ~s({"query":"elixir"}),
                    "builtin?" => true
                  }
                }
              ]
            }
          ]
        }
      ]

      for context <- contexts do
        request = build_request(context: context)
        encoded = ResponsesAPI.encode_body(request)
        body = Jason.decode!(encoded.body)

        refute Enum.any?(body["input"], &(&1["type"] == "function_call"))
      end
    end

    test "requires and encodes explicit results for provider-native calls" do
      base_context = ReqLLM.Context.new([ReqLLM.Context.user("Search provider data")])

      provider_native_call =
        "native_1"
        |> ReqLLM.ToolCall.new("provider_search", ~s({"query":"elixir"}))
        |> ReqLLM.ToolCall.put_metadata(%{provider_native: :openai})

      assistant =
        ReqLLM.Context.assistant("", tool_calls: [provider_native_call])

      assert {:error, %ReqLLM.Error.Validation.Error{context: error_context}} =
               ReqLLM.Context.append_tool_exchange(base_context, assistant, [])

      assert error_context[:kind] == :missing_tool_results

      result =
        ReqLLM.Context.tool_result(
          "native_1",
          "provider_search",
          "provider search complete"
        )

      assert {:ok, context} =
               ReqLLM.Context.append_tool_exchange(base_context, assistant, [result])

      request = build_request(context: context)
      body = request |> ResponsesAPI.encode_body() |> ReqLLM.Test.Helpers.json_body()

      function_items =
        Enum.filter(body["input"], &(&1["type"] in ["function_call", "function_call_output"]))

      assert Enum.map(function_items, & &1["type"]) == [
               "function_call",
               "function_call_output"
             ]

      assert Enum.map(function_items, & &1["call_id"]) == ["native_1", "native_1"]
    end

    test "encodes multimodal tool results as array function_call_output" do
      tool_call = %ReqLLM.ToolCall{
        id: "call_1",
        type: "function",
        function: %{name: "take_screenshot", arguments: ~s({})}
      }

      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [],
        tool_calls: [tool_call]
      }

      tool_result =
        ReqLLM.Context.tool_result("call_1", [
          ReqLLM.Message.ContentPart.text("Captured screenshot"),
          ReqLLM.Message.ContentPart.image(<<137, 80, 78, 71>>, "image/png")
        ])

      context = %ReqLLM.Context{messages: [assistant_msg, tool_result]}
      request = build_request(context: context)

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      tool_output =
        Enum.find(body["input"], fn item ->
          item["type"] == "function_call_output"
        end)

      assert tool_output["call_id"] == "call_1"
      assert is_list(tool_output["output"])

      assert Enum.any?(tool_output["output"], fn part ->
               part["type"] == "input_text" and part["text"] == "Captured screenshot"
             end)

      assert Enum.any?(tool_output["output"], fn part ->
               part["type"] == "input_image" and
                 String.starts_with?(part["image_url"], "data:image/png;base64,")
             end)
    end

    test "encodes file-bearing tool results as array function_call_output" do
      tool_call = %ReqLLM.ToolCall{
        id: "call_1",
        type: "function",
        function: %{name: "export_report", arguments: ~s({})}
      }

      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [],
        tool_calls: [tool_call]
      }

      tool_result =
        ReqLLM.Context.tool_result("call_1", [
          ReqLLM.Message.ContentPart.text("Attached report"),
          ReqLLM.Message.ContentPart.file("report-bytes", "report.txt", "text/plain")
        ])

      context = %ReqLLM.Context{messages: [assistant_msg, tool_result]}
      request = build_request(context: context)

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      tool_output =
        Enum.find(body["input"], fn item ->
          item["type"] == "function_call_output"
        end)

      assert tool_output["call_id"] == "call_1"
      assert is_list(tool_output["output"])

      assert Enum.any?(tool_output["output"], fn part ->
               part["type"] == "input_text" and part["text"] == "Attached report"
             end)

      assert Enum.any?(tool_output["output"], fn part ->
               part["type"] == "input_file" and
                 part["filename"] == "report.txt" and
                 String.starts_with?(part["file_data"], "data:text/plain;base64,")
             end)
    end

    test "encodes canonical specific tool choice with atom keys" do
      request = build_request(tool_choice: %{type: "tool", name: "get_weather"})

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert body["tool_choice"] == %{"type" => "function", "name" => "get_weather"}
    end

    test "encodes OpenAI Chat-style specific tool choice with atom keys" do
      request =
        build_request(tool_choice: %{type: "function", function: %{name: "get_weather"}})

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert body["tool_choice"] == %{"type" => "function", "name" => "get_weather"}
    end

    test "encodes OpenAI Chat-style specific tool choice with string keys" do
      request =
        build_request(tool_choice: %{"type" => "function", "function" => %{"name" => "search"}})

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert body["tool_choice"] == %{"type" => "function", "name" => "search"}
    end

    test "encodes parallel_tool_calls true" do
      request = build_request(parallel_tool_calls: true)

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert body["parallel_tool_calls"] == true
    end

    test "encodes parallel_tool_calls false" do
      request = build_request(parallel_tool_calls: false)

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert body["parallel_tool_calls"] == false
    end

    test "omits parallel_tool_calls when not set" do
      request = build_request([])

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      refute Map.has_key?(body, "parallel_tool_calls")
    end
  end

  describe "tool search items" do
    @search_call %{
      "arguments" => %{"paths" => ["devices"]},
      "call_id" => nil,
      "execution" => "server",
      "id" => "tsc_123",
      "status" => "completed",
      "type" => "tool_search_call"
    }

    @search_output %{
      "call_id" => nil,
      "execution" => "server",
      "id" => "tso_123",
      "status" => "completed",
      "tools" => [
        %{
          "type" => "namespace",
          "name" => "devices",
          "description" => "Device tools.",
          "tools" => [
            %{
              "defer_loading" => true,
              "description" => "List managed devices at a site.",
              "name" => "list",
              "output_schema" => nil,
              "parameters" => %{
                "additionalProperties" => false,
                "properties" => %{"site" => %{"type" => "string"}},
                "required" => ["site"],
                "type" => "object"
              },
              "strict" => true,
              "type" => "function"
            }
          ]
        }
      ],
      "type" => "tool_search_output"
    }

    @namespaced_call %{
      "arguments" => ~s({"site":"hq"}),
      "call_id" => "call_123",
      "id" => "fc_123",
      "name" => "list",
      "namespace" => "devices",
      "status" => "completed",
      "type" => "function_call"
    }

    @commentary %{
      "type" => "message",
      "role" => "assistant",
      "content" => [%{"type" => "output_text", "text" => "Looking up the device tools."}]
    }

    defp search_response do
      %{
        "id" => "resp_123",
        "model" => "gpt-5.4",
        "status" => "completed",
        "output" => [@commentary, @search_call, @search_output, @namespaced_call],
        "usage" => %{"input_tokens" => 10, "output_tokens" => 5}
      }
    end

    defp list_tool(provider_options) do
      ReqLLM.Tool.new!(
        name: "list",
        description: "List managed devices at a site.",
        parameter_schema: [site: [type: :string, required: true]],
        callback: fn _ -> {:ok, "17"} end,
        provider_options: provider_options
      )
    end

    defp encode_tools(tools) do
      context = ReqLLM.Context.new([ReqLLM.Context.user("hi")])

      %Req.Request{
        url: URI.parse("https://api.openai.com/v1/responses"),
        method: :post,
        options: [context: context, model: "gpt-5.4", tools: tools]
      }
      |> ResponsesAPI.encode_body()
      |> ReqLLM.Test.Helpers.json_body()
      |> Map.fetch!("tools")
    end

    test "decode_response/1 keeps search items as provider blocks in output order" do
      {_req, resp} = ResponsesAPI.decode_response(build_response(200, search_response()))

      assert [
               %ReqLLM.Message.ContentPart{type: :text, text: "Looking up the device tools."},
               %ReqLLM.Message.ContentPart{type: :provider_block, data: @search_call} = call_part,
               %ReqLLM.Message.ContentPart{type: :provider_block, data: @search_output}
             ] = resp.body.message.content

      assert call_part.metadata.provider == :openai
      assert call_part.metadata.block_type == "tool_search_call"
      refute Map.has_key?(resp.body.message.metadata, :responses_replay)
    end

    test "decode_response/1 keeps a function call's namespace in its metadata" do
      {_req, resp} = ResponsesAPI.decode_response(build_response(200, search_response()))

      assert [tool_call] = resp.body.message.tool_calls
      assert ReqLLM.ToolCall.name(tool_call) == "list"
      assert ReqLLM.ToolCall.metadata(tool_call) == %{namespace: "devices"}
      assert resp.body.finish_reason == :tool_calls
    end

    test "a namespaced call still resolves as an application tool call" do
      {_req, resp} = ResponsesAPI.decode_response(build_response(200, search_response()))
      [tool_call] = resp.body.message.tool_calls

      assert %{state: :valid, arguments: %{"site" => "hq"}} =
               ReqLLM.ToolCall.resolve(tool_call, [list_tool(openai: [namespace: "devices"])])
    end

    test "a decoded namespaced call executes the matching tool when names repeat" do
      {_req, resp} = ResponsesAPI.decode_response(build_response(200, search_response()))
      [tool_call] = resp.body.message.tool_calls

      billing = list_tool(openai: [namespace: "billing"])
      billing = %{billing | callback: fn _ -> flunk("executed the billing tool") end}
      devices = list_tool(openai: [namespace: "devices"])

      assert {:ok, "17"} = ReqLLM.ToolCall.execute(tool_call, [billing, devices])
    end

    test "streamed search items and the call's namespace match the decoded response" do
      {:ok, model} = ReqLLM.model("openai:gpt-5.4")

      events = [
        %{
          "type" => "response.output_item.added",
          "output_index" => 0,
          "item" => %{@search_call | "arguments" => %{}, "status" => "in_progress"}
        },
        %{"type" => "response.output_item.done", "output_index" => 0, "item" => @search_call},
        %{
          "type" => "response.output_item.added",
          "output_index" => 1,
          "item" => %{@search_output | "status" => "in_progress"}
        },
        %{"type" => "response.output_item.done", "output_index" => 1, "item" => @search_output},
        %{
          "type" => "response.output_item.added",
          "output_index" => 2,
          "item" => %{@namespaced_call | "arguments" => "", "status" => "in_progress"}
        },
        %{
          "type" => "response.function_call_arguments.delta",
          "output_index" => 2,
          "item_id" => "fc_123",
          "delta" => ~s({"site":"hq"})
        },
        %{
          "type" => "response.function_call_arguments.done",
          "output_index" => 2,
          "item_id" => "fc_123",
          "arguments" => ~s({"site":"hq"})
        },
        %{"type" => "response.output_item.done", "output_index" => 2, "item" => @namespaced_call}
      ]

      {chunks, _state} =
        Enum.flat_map_reduce(events, ResponsesAPI.init_stream_state(), fn event, state ->
          ResponsesAPI.decode_stream_event(%{data: event}, model, state)
        end)

      {:ok, response} =
        ReqLLM.Provider.ResponseBuilder.for_model(model).build_response(
          chunks,
          %{id: "resp_123", model: "gpt-5.4", finish_reason: :tool_calls},
          context: ReqLLM.Context.new([]),
          model: model
        )

      assert [
               %ReqLLM.Message.ContentPart{type: :provider_block, data: @search_call},
               %ReqLLM.Message.ContentPart{type: :provider_block, data: @search_output}
             ] = response.message.content

      assert [tool_call] = response.message.tool_calls
      assert ReqLLM.ToolCall.args_map(tool_call) == %{"site" => "hq"}
      assert ReqLLM.ToolCall.metadata(tool_call) == %{namespace: "devices"}
    end

    test "encode_body/1 replays a decoded search turn as the items the API returned" do
      {_req, resp} = ResponsesAPI.decode_response(build_response(200, search_response()))

      context =
        ReqLLM.Context.new([
          ReqLLM.Context.user("How many devices at hq?"),
          resp.body.message,
          ReqLLM.Context.tool_result("call_123", "17 devices.")
        ])

      body =
        build_request(context: context, provider_options: [store: false])
        |> ResponsesAPI.encode_body()
        |> ReqLLM.Test.Helpers.json_body()

      assert [user, commentary, search_call, search_output, function_call, output] =
               body["input"]

      assert user["role"] == "user"
      assert commentary["role"] == "assistant"
      assert search_call == @search_call
      assert search_output == @search_output

      assert function_call == %{
               "type" => "function_call",
               "call_id" => "call_123",
               "name" => "list",
               "namespace" => "devices",
               "arguments" => ~s({"site":"hq"})
             }

      assert output["type"] == "function_call_output"
    end

    test "encode_body/1 sends a call without a namespace without the key" do
      tool_call = ReqLLM.ToolCall.new("call_1", "get_time", "{}")

      context =
        ReqLLM.Context.new([
          ReqLLM.Context.user("Time?"),
          ReqLLM.Context.assistant("", tool_calls: [tool_call])
        ])

      body =
        build_request(context: context)
        |> ResponsesAPI.encode_body()
        |> ReqLLM.Test.Helpers.json_body()

      assert [_user, function_call] = body["input"]
      refute Map.has_key?(function_call, "namespace")
    end

    test "encode_body/1 drops search items owned by another provider" do
      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [
          ReqLLM.Message.ContentPart.provider_block(:anthropic, @search_call),
          %ReqLLM.Message.ContentPart{type: :text, text: "Answer"}
        ]
      }

      body =
        build_request(context: ReqLLM.Context.new([assistant_msg]))
        |> ResponsesAPI.encode_body()
        |> ReqLLM.Test.Helpers.json_body()

      refute Enum.any?(body["input"], &(&1["type"] in ["tool_search_call", "tool_search_output"]))
    end

    test "encode_body/1 skips search items when chaining via previous_response_id" do
      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [
          ReqLLM.Message.ContentPart.provider_block(:openai, @search_call),
          ReqLLM.Message.ContentPart.provider_block(:openai, @search_output)
        ],
        metadata: %{response_id: "resp_prev"}
      }

      body =
        build_request(context: ReqLLM.Context.new([assistant_msg, ReqLLM.Context.user("Next")]))
        |> ResponsesAPI.encode_body()
        |> ReqLLM.Test.Helpers.json_body()

      assert body["previous_response_id"] == "resp_prev"
      refute Enum.any?(body["input"], &(&1["type"] in ["tool_search_call", "tool_search_output"]))
    end

    test "functions sharing a namespace are sent as one namespace tool in place of the first" do
      direct =
        ReqLLM.Tool.new!(
          name: "get_time",
          description: "Get the time.",
          parameter_schema: [],
          callback: fn _ -> {:ok, "now"} end
        )

      restart =
        ReqLLM.Tool.new!(
          name: "restart",
          description: "Restart a device.",
          parameter_schema: [id: [type: :string, required: true]],
          callback: fn _ -> {:ok, "ok"} end,
          provider_options: [
            openai: [
              defer_loading: true,
              namespace: "devices",
              namespace_description: "Inspect and manage devices."
            ]
          ]
        )

      invoices =
        ReqLLM.Tool.new!(
          name: "invoices",
          description: "List invoices.",
          parameter_schema: [],
          callback: fn _ -> {:ok, "none"} end,
          provider_options: %{openai: %{"defer_loading" => true, "namespace" => "billing"}}
        )

      tools = [
        direct,
        list_tool(openai: [defer_loading: true, namespace: "devices"]),
        %{type: "tool_search"},
        invoices,
        restart
      ]

      assert [
               %{"type" => "function", "name" => "get_time"},
               %{
                 "type" => "namespace",
                 "name" => "devices",
                 "description" => "Inspect and manage devices.",
                 "tools" => [
                   %{"type" => "function", "name" => "list", "defer_loading" => true},
                   %{"type" => "function", "name" => "restart", "defer_loading" => true}
                 ]
               },
               %{"type" => "tool_search"},
               %{
                 "type" => "namespace",
                 "name" => "billing",
                 "description" => "Tools for billing.",
                 "tools" => [%{"type" => "function", "name" => "invoices"}]
               }
             ] = encode_tools(tools)
    end
  end
end
