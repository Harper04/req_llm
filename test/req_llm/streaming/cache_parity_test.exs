defmodule ReqLLM.Streaming.CacheParityTest do
  use ExUnit.Case, async: true

  alias ReqLLM.{Context, Message, Response, StreamChunk, StreamResponse}

  @schema [count: [type: :integer, required: true]]

  defmodule Provider do
    use ReqLLM.Provider,
      id: :stream_cache_parity,
      default_base_url: "in-process://stream-cache-parity"

    def stream_transport(_model, opts) do
      send(opts[:owner], :transport_selected)
      :in_process
    end

    def prepare_request(:object, _model, messages, opts) do
      send(opts[:owner], {:prepared_object, opts})
      {:ok, context} = Context.normalize(messages, opts)
      opts = Keyword.put(opts, :context, context)

      request =
        Req.new()
        |> Req.Request.register_options(Keyword.keys(opts))
        |> Req.merge(opts)

      {:ok, request}
    end

    def attach_in_process_stream(model, context, opts) do
      send(opts[:owner], {:attached, model, context, opts})

      {:ok,
       [
         StreamChunk.text(~s({"count":7})),
         StreamChunk.meta(%{
           terminal?: true,
           finish_reason: :stop,
           usage: %{input_tokens: 2, output_tokens: 3, total_tokens: 5}
         })
       ]}
    end
  end

  defmodule Backend do
    @behaviour ReqLLM.Cache

    def generate_key(model, request, opts) do
      send(opts[:owner], {:cache_key, model.id, request})
      {model.id, request.operation, request.context, request.schema}
    end

    def get(key, opts) do
      send(opts[:owner], {:cache_get, key})
      Keyword.fetch!(opts, :result)
    end

    def put(_key, _value, _ttl, opts) do
      send(opts[:owner], :cache_put)
      :ok
    end

    def delete(_key, _opts), do: :ok
  end

  setup_all do
    assert {:ok, :stream_cache_parity} = ReqLLM.Providers.register(Provider)
    on_exit(fn -> ReqLLM.Providers.unregister(:stream_cache_parity) end)
  end

  setup do
    %{model: %{provider: :stream_cache_parity, id: "outside-catalog"}}
  end

  test "text cache hits replay without selecting a transport, starting a provider or a deadline",
       %{
         model: model
       } do
    cached = %Response{
      id: "cached-response",
      model: "outside-catalog",
      message: %Message{role: :assistant, content: [ReqLLM.Message.ContentPart.text("cached")]},
      context: Context.new("Hello"),
      finish_reason: :stop,
      usage: %{input_tokens: 8, output_tokens: 9, total_tokens: 17},
      provider_meta: %{}
    }

    assert {:ok, response} =
             ReqLLM.stream_text(model, "Hello",
               owner: self(),
               total_timeout: :must_not_validate_on_hit,
               cache: Backend,
               cache_options: [owner: self(), result: {:ok, cached}]
             )

    assert_receive {:cache_key, "outside-catalog", %{operation: :chat, schema: nil}}
    assert_receive {:cache_get, _key}
    refute_received :transport_selected
    refute_received {:attached, _, _, _}
    assert StreamResponse.text(response) == "cached"
    assert StreamResponse.usage(response).total_tokens == 0
    assert StreamResponse.finish_reason(response) == :stop
  end

  test "text cache misses start the stream without caching its result", %{model: model} do
    schema = NimbleOptions.new!(value: [type: :string])

    assert {:ok, response} =
             ReqLLM.stream_text(model, "Hello",
               owner: self(),
               cache: Backend,
               cache_options: [owner: self(), result: {:error, :not_found}],
               extension_schema: schema,
               provider_options: %{stream_cache_parity: %{opaque: "preserved"}}
             )

    assert_receive {:cache_key, "outside-catalog", %{operation: :chat, schema: nil}}
    assert_receive {:cache_get, _key}
    assert_receive :transport_selected
    assert_receive {:attached, %LLMDB.Model{id: "outside-catalog"}, %Context{}, opts}
    assert opts[:extension_schema] === schema
    assert opts[:provider_options] == [opaque: "preserved"]
    refute Keyword.has_key?(opts, :cache)
    refute Keyword.has_key?(opts, :cache_options)
    assert StreamResponse.text(response) == ~s({"count":7})
    assert StreamResponse.usage(response).total_tokens == 5
    refute_received :cache_put
  end

  for result <- [:hit, :miss] do
    @result result

    test "object streaming keeps its existing no-cache path even with a cache #{@result}", %{
      model: model
    } do
      cache_result =
        if @result == :hit,
          do:
            {:ok,
             %Response{
               id: "cached",
               model: "outside-catalog",
               message: nil,
               context: Context.new()
             }},
          else: {:error, :not_found}

      assert {:ok, response} =
               ReqLLM.stream_object(model, "Hello", @schema,
                 owner: self(),
                 cache: Backend,
                 cache_options: [owner: self(), result: cache_result]
               )

      assert_receive {:prepared_object, prepared_opts}
      assert prepared_opts[:cache] == Backend
      assert_receive :transport_selected
      assert_receive {:attached, _, %Context{}, opts}
      assert opts[:operation] == :object
      assert %{compiled: %NimbleOptions{schema: [count: count_opts]}} = opts[:compiled_schema]
      assert count_opts[:type] == :integer
      assert {:ok, %Response{object: %{"count" => 7}}} = StreamResponse.to_response(response)
      refute_received {:cache_key, _, _}
      refute_received {:cache_get, _}
      refute_received :cache_put
    end
  end
end
