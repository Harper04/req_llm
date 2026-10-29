defmodule ReqLLM.GenerationRuntimeTest do
  use ExUnit.Case, async: true

  alias ReqLLM.Context
  alias ReqLLM.Generation
  alias ReqLLM.Message
  alias ReqLLM.Response

  @moduletag contract: :public_api
  @schema [count: [type: :integer, required: true]]

  defmodule Provider do
    @behaviour ReqLLM.Provider

    def provider_id, do: :generation_runtime_parity

    def prepare_request(operation, model, context, opts) do
      send(opts[:owner], {:prepared, operation, model, context, opts})
      Process.sleep(Keyword.get(opts, :prepare_delay, 0))
      Keyword.get(opts, :prepare_result, {:ok, opts[:request]})
    end

    def attach(request, _model, _opts), do: request
    def encode_body(request), do: request
    def decode_response(result), do: result
  end

  defmodule CacheBackend do
    @behaviour ReqLLM.Cache

    def generate_key(model, request, opts) do
      send(opts[:owner], {:cache_key, model, request})
      {model.id, request.operation, request.context, request.schema}
    end

    def get(key, opts) do
      case :ets.lookup(opts[:table], key) do
        [{^key, value}] -> {:ok, value}
        [] -> {:error, :not_found}
      end
    end

    def put(key, value, ttl, opts) do
      send(opts[:owner], {:stored, key, value, ttl, opts})
      :ets.insert(opts[:table], {key, value})
      :ok
    end

    def delete(key, opts) do
      :ets.delete(opts[:table], key)
      :ok
    end
  end

  setup_all do
    assert {:ok, :generation_runtime_parity} = ReqLLM.Providers.register(Provider)
    on_exit(fn -> ReqLLM.Providers.unregister(:generation_runtime_parity) end)
    :ok
  end

  setup do
    table = :ets.new(__MODULE__, [:set, :public])
    model = %{provider: :generation_runtime_parity, id: "outside-the-catalog"}
    %{model: model, cache_options: [table: table, owner: self()]}
  end

  for operation <- [:chat, :object] do
    @operation operation

    test "#{operation} caches the decoded result and skips preparation and the deadline on a hit",
         %{
           model: model,
           cache_options: cache_options
         } do
      operation = @operation
      request = decoded_request(response_fixture(), self())
      pricing_context = %{pricing_period: "peak"}
      extension_schema = NimbleOptions.new!(value: [type: :string])

      opts = [
        owner: self(),
        request: request,
        pricing_context: pricing_context,
        extension_schema: extension_schema,
        provider_options: %{generation_runtime_parity: %{opaque: "value"}},
        system_prompt: "Be precise",
        cache: CacheBackend,
        cache_ttl: 300,
        cache_options: cache_options
      ]

      assert {:ok, first} = generate(operation, model, opts)

      assert_receive {:prepared, ^operation, %LLMDB.Model{id: "outside-the-catalog"}, context,
                      prepared_opts}

      assert %Context{messages: [%Message{role: :system}, %Message{role: :user}]} = context
      assert prepared_opts[:extension_schema] === extension_schema
      assert prepared_opts[:provider_options] == [opaque: "value"]
      refute Keyword.has_key?(prepared_opts, :pricing_context)
      refute Keyword.has_key?(prepared_opts, :cache)
      refute Keyword.has_key?(prepared_opts, :cache_options)
      refute Keyword.has_key?(prepared_opts, :cache_ttl)
      assert_receive {:sent, %Req.Request{private: %{req_llm_pricing_context: ^pricing_context}}}

      assert_receive {:cache_key, _,
                      %{operation: ^operation, context: ^context, schema: key_schema}}

      if operation == :object do
        assert key_schema == @schema
        assert %{schema: @schema, compiled: %NimbleOptions{}} = prepared_opts[:compiled_schema]
        assert first.object == %{"count" => 42}
      else
        assert key_schema == nil
        refute Keyword.has_key?(prepared_opts, :compiled_schema)
        assert Response.text(first) == "Hello"
      end

      assert_receive {:stored, key, ^first, 300, ^cache_options}
      assert first.usage.total_tokens == 7

      hit_opts =
        opts
        |> Keyword.put(:prepare_result, {:error, :must_not_prepare})
        |> Keyword.put(:total_timeout, :invalid_on_purpose)

      assert {:ok, cached} = generate(operation, model, hit_opts)
      assert cached.message == first.message
      assert cached.object == first.object
      assert cached.usage.total_tokens == 0
      assert cached.usage.input_tokens == 0
      assert cached.usage.output_tokens == 0
      assert cached.provider_meta.response_cache_hit == true
      assert %Context{messages: [_, _, %Message{role: :assistant}]} = cached.context
      assert :ets.lookup(cache_options[:table], key) == [{key, first}]
      refute_received {:prepared, _, _, _, _}
      refute_received {:sent, _}
      refute_received {:stored, _, _, _, _}
    end

    test "#{operation} preserves preparation errors without sending or caching", %{
      model: model,
      cache_options: cache_options
    } do
      error = {:provider_rejected, make_ref()}

      assert {:error, ^error} =
               generate(@operation, model,
                 owner: self(),
                 request: decoded_request(response_fixture(), self()),
                 prepare_result: {:error, error},
                 cache: CacheBackend,
                 cache_options: cache_options
               )

      refute_received {:sent, _}
      refute_received {:stored, _, _, _, _}
    end

    test "#{operation} preserves completed HTTP errors and does not cache them", %{
      model: model,
      cache_options: cache_options
    } do
      request =
        Req.new(
          url: "https://generation.example",
          retry: false,
          plug: fn conn -> Plug.Conn.send_resp(conn, 418, "provider error body") end
        )

      assert {:error, %ReqLLM.Error.API.Request{} = error} =
               generate(@operation, model,
                 owner: self(),
                 request: request,
                 cache: CacheBackend,
                 cache_options: cache_options
               )

      assert error.reason == "HTTP 418: Request failed"
      assert error.status == 418
      assert error.response_body == "provider error body"
      assert Exception.message(error) == "API request failed (418): HTTP 418: Request failed"
      refute_received {:stored, _, _, _, _}
    end

    test "#{operation} preserves transport errors without caching", %{
      model: model,
      cache_options: cache_options
    } do
      request =
        Req.new(
          url: "https://generation.example",
          retry: false,
          plug: fn conn -> Req.Test.transport_error(conn, :econnrefused) end
        )

      assert {:error, %Req.TransportError{reason: :econnrefused}} =
               generate(@operation, model,
                 owner: self(),
                 request: request,
                 cache: CacheBackend,
                 cache_options: cache_options
               )

      refute_received {:stored, _, _, _, _}
    end

    test "#{operation} includes provider preparation in the cache-miss deadline", %{
      model: model,
      cache_options: cache_options
    } do
      assert {:error, %ReqLLM.Error.API.Timeout{kind: :total, timeout: 1}} =
               generate(@operation, model,
                 owner: self(),
                 request: decoded_request(response_fixture(), self()),
                 prepare_delay: 20,
                 total_timeout: 1,
                 cache: CacheBackend,
                 cache_options: cache_options
               )

      refute_received {:sent, _}
      refute_received {:stored, _, _, _, _}
    end
  end

  for operation <- [:chat, :object] do
    @operation operation

    test "#{operation} retains provider decoding, pricing and request telemetry" do
      operation = @operation
      owner = self()
      conversation_id = "buffered-parity-#{System.unique_integer([:positive])}"
      handler_id = {__MODULE__, make_ref()}
      stub = {__MODULE__, make_ref()}

      :ok =
        :telemetry.attach_many(
          handler_id,
          [
            [:req_llm, :request, :start],
            [:req_llm, :request, :stop],
            [:req_llm, :request, :exception]
          ],
          fn event, measurements, metadata, _config ->
            if get_in(metadata, [:request_options, :conversation_id]) == conversation_id do
              send(owner, {:telemetry, event, measurements, metadata})
            end
          end,
          nil
        )

      on_exit(fn -> :telemetry.detach(handler_id) end)

      Req.Test.stub(stub, fn conn ->
        refute Map.has_key?(conn.body_params, "pricing_context")

        message =
          if operation == :object do
            %{
              "role" => "assistant",
              "tool_calls" => [
                %{
                  "id" => "call-parity",
                  "type" => "function",
                  "function" => %{
                    "name" => "structured_output",
                    "arguments" => ~s({"count":42})
                  }
                }
              ]
            }
          else
            %{"role" => "assistant", "content" => "Hello"}
          end

        Req.Test.json(conn, %{
          "id" => "provider-parity",
          "model" => "gpt-4-turbo",
          "choices" => [%{"message" => message, "finish_reason" => "stop"}],
          "usage" => %{"prompt_tokens" => 5, "completion_tokens" => 2, "total_tokens" => 7}
        })
      end)

      model = %LLMDB.Model{
        provider: :openai,
        id: "gpt-4-turbo",
        pricing: %{
          currency: "USD",
          components: [
            %{
              id: "token.input",
              kind: "token",
              rate: 1.0,
              per: 1_000_000,
              applies_when: %{pricing_period: "peak"}
            },
            %{id: "token.output", kind: "token", rate: 2.0, per: 1_000_000}
          ]
        }
      }

      assert {:ok, response} =
               generate(operation, model,
                 api_key: "test-key",
                 pricing_context: %{pricing_period: "peak"},
                 telemetry: [conversation_id: conversation_id],
                 req_http_options: [plug: {Req.Test, stub}]
               )

      assert response.id == "provider-parity"
      assert response.usage.pricing.status == :priced
      assert_in_delta response.usage.total_cost, 0.000009, 0.000000001

      if operation == :object do
        assert Response.object(response) == %{"count" => 42}
      else
        assert Response.text(response) == "Hello"
      end

      assert_receive {:telemetry, [:req_llm, :request, :start], %{system_time: system_time},
                      start}

      assert_receive {:telemetry, [:req_llm, :request, :stop], %{duration: duration}, stop}
      assert is_integer(system_time)
      assert duration >= 0
      assert start.operation == operation
      assert start.provider == :openai
      assert start.mode == :sync
      assert start.transport == :req
      assert %LLMDB.Model{provider: :openai, id: "gpt-4-turbo"} = start.model
      assert start.request_options.conversation_id == conversation_id
      assert stop.request_id == start.request_id
      assert stop.http_status == 200
      assert stop.usage.tokens.total_tokens == 7
      refute_received {:telemetry, [:req_llm, :request, :exception], _, _}
      refute_received {:telemetry, [:req_llm, :request, :start], _, _}
      refute_received {:telemetry, [:req_llm, :request, :stop], _, _}
    end
  end

  test "object cache keys distinguish schemas for the same normalized context", %{
    model: model,
    cache_options: cache_options
  } do
    opts = [
      owner: self(),
      request: decoded_request(response_fixture(), self()),
      cache: CacheBackend,
      cache_options: cache_options
    ]

    assert {:ok, _} = Generation.generate_object(model, "Hello", @schema, opts)
    other_schema = [count: [type: :string, required: true]]
    assert {:ok, other} = Generation.generate_object(model, "Hello", other_schema, opts)
    assert other.object == %{"count" => "42"}
    assert :ets.info(cache_options[:table], :size) == 2
    assert_receive {:prepared, :object, _, _, _}
    assert_receive {:prepared, :object, _, _, _}
  end

  test "json_strict models bypass coercion before caching", %{
    model: model,
    cache_options: cache_options
  } do
    model = Map.put(model, :capabilities, %{json: %{strict: true}})

    assert {:ok, response} =
             generate(:object, model,
               owner: self(),
               request: decoded_request(response_fixture(), self()),
               cache: CacheBackend,
               cache_options: cache_options
             )

    assert response.object == %{"count" => "42"}
    assert_receive {:stored, _, ^response, _, _}
  end

  defp generate(:chat, model, opts), do: Generation.generate_text(model, "Hello", opts)

  defp generate(:object, model, opts),
    do: Generation.generate_object(model, "Hello", @schema, opts)

  defp decoded_request(response, owner) do
    Req.new(
      url: "https://generation.example",
      plug: fn conn -> Plug.Conn.send_resp(conn, 200, "wire response") end
    )
    |> Req.Request.append_request_steps(
      observe: fn request ->
        send(owner, {:sent, request})
        request
      end
    )
    |> Req.Request.append_response_steps(
      decode: fn {request, raw_response} ->
        {request, %{raw_response | body: response}}
      end
    )
  end

  defp response_fixture do
    %Response{
      id: "generation-parity",
      model: "outside-the-catalog",
      message: Context.assistant("Hello"),
      context: Context.new([Context.user("Hello"), Context.assistant("Hello")]),
      object: %{"count" => "42"},
      usage: %{input_tokens: 5, output_tokens: 2, total_tokens: 7},
      finish_reason: :stop,
      provider_meta: %{source: :provider}
    }
  end
end
