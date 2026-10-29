defmodule ReqLLM.Provider.AdapterLoadingTest do
  use ExUnit.Case, async: false

  alias ReqLLM.Provider.Adapter

  @moduletag contract: :provider_extension

  @tag :tmp_dir
  test "optional detection does not load a provider while direct entry points still autoload", %{
    tmp_dir: tmp_dir
  } do
    provider = Module.concat(__MODULE__, ColdProvider)
    install_unloaded_provider(provider, tmp_dir)
    model = %LLMDB.Model{provider: :external_adapter, id: "cold-model"}
    {:ok, context} = ReqLLM.Context.normalize("Hello")
    request = Req.Request.put_private(Req.new(), :opaque, make_ref())
    finch_request = Finch.build(:post, "https://adapter.example/cold")
    parser = &ReqLLM.Streaming.SSE.accumulate_and_parse/2
    chunks = [ReqLLM.StreamChunk.text("Hello")]
    websocket = %{url: "wss://adapter.example/cold", headers: []}

    opts = [
      prepared_result: {:ok, request},
      stream_result: {:ok, finch_request},
      parser: parser,
      chunks: chunks,
      websocket: websocket
    ]

    assert :code.is_loaded(provider) == false
    refute function_exported?(provider, :stream_transport, 2)
    refute function_exported?(provider, :stream_protocol_parser, 2)
    refute function_exported?(provider, :attach_in_process_stream, 3)
    refute function_exported?(provider, :attach_websocket_stream, 3)

    assert Adapter.stream_transport(provider, model, opts) == :http
    assert Adapter.stream_protocol_parser(provider, model, opts) == nil

    assert Adapter.attach_in_process_stream(provider, model, context, opts) ==
             {:error, {:missing_callback, :attach_in_process_stream, 3}}

    assert Adapter.attach_websocket_stream(provider, model, context, opts) == false
    assert :code.is_loaded(provider) == false

    assert provider.prepare_request(:ocr, model, "document", opts) === {:ok, request}
    unload(provider)
    assert Adapter.prepare_request(provider, :ocr, model, "document", opts) === {:ok, request}

    assert provider.attach_stream(model, context, opts, ColdFinch) === {:ok, finch_request}
    unload(provider)

    assert Adapter.attach_stream(provider, model, context, opts, ColdFinch) ===
             {:ok, finch_request}

    assert Adapter.stream_transport(provider, model, opts) ===
             provider.stream_transport(model, opts)

    assert Adapter.stream_transport(provider, model, opts) == :websocket
    assert Adapter.stream_protocol_parser(provider, model, opts) === parser

    assert Adapter.stream_protocol_parser(provider, model, opts) ===
             provider.stream_protocol_parser(model, opts)

    assert Adapter.attach_in_process_stream(provider, model, context, opts) === {:ok, chunks}

    assert Adapter.attach_in_process_stream(provider, model, context, opts) ===
             provider.attach_in_process_stream(model, context, opts)

    assert Adapter.attach_websocket_stream(provider, model, context, opts) === {:ok, websocket}

    assert Adapter.attach_websocket_stream(provider, model, context, opts) ===
             provider.attach_websocket_stream(model, context, opts)
  end

  defp install_unloaded_provider(provider, tmp_dir) do
    [{^provider, beam}] =
      Code.compile_quoted(
        quote do
          defmodule unquote(provider) do
            @behaviour ReqLLM.Provider

            def prepare_request(_operation, _model, _input, opts), do: opts[:prepared_result]
            def attach(request, _model, _opts), do: request
            def encode_body(request), do: request
            def decode_response(request_response), do: request_response
            def stream_transport(_model, _opts), do: :websocket
            def stream_protocol_parser(_model, opts), do: opts[:parser]
            def attach_stream(_model, _context, opts, _finch_name), do: opts[:stream_result]
            def attach_in_process_stream(_model, _context, opts), do: {:ok, opts[:chunks]}
            def attach_websocket_stream(_model, _context, opts), do: {:ok, opts[:websocket]}
          end
        end
      )

    File.write!(Path.join(tmp_dir, "#{provider}.beam"), beam)
    Code.prepend_path(tmp_dir)
    unload(provider)

    on_exit(fn ->
      unload(provider)
      Code.delete_path(tmp_dir)
    end)
  end

  defp unload(provider) do
    :code.purge(provider)
    :code.delete(provider)
  end
end
