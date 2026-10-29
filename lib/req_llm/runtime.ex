defmodule ReqLLM.Runtime do
  @moduledoc false

  alias ReqLLM.Provider.Adapter
  alias ReqLLM.RequestPlan
  alias ReqLLM.TimeoutBudget
  alias ReqLLM.StreamServer

  require Logger

  @doc """
  Executes a synchronous Req plan using the existing provider and timeout paths.

  The operation boundary resolves the model and provider, processes options, and
  starts the deadline before building the plan. The adapter prepares the provider's
  Req pipeline unchanged; TimeoutBudget attaches the current transport and runs it.
  Preparation and transport errors pass through without invoking the result policy.

  The result policy is a single function owned by the operation. It receives the
  completed Req response, including non-2xx responses, and retains the operation's
  current normalization and HTTP error handling. It runs outside the request budget,
  as it does at the existing operation boundaries.
  """
  @spec execute(
          RequestPlan.t(),
          term(),
          TimeoutBudget.deadline(),
          (Req.Response.t() -> {:ok, term()} | {:error, term()})
        ) :: {:ok, term()} | {:error, term()}
  def execute(%RequestPlan{transport: :req} = plan, input, deadline, result_policy)
      when is_function(result_policy, 1) do
    with {:ok, request} <- prepare(plan, input) do
      execute_prepared(request, deadline, result_policy)
    end
  end

  @doc """
  Prepares a synchronous Req plan without sending it.

  The operation can inspect or transform the provider's request before execution.
  The existing provider callback receives the plan's model, input, and options
  unchanged, and preparation errors pass through unchanged.
  """
  @spec prepare(RequestPlan.t(), term()) :: {:ok, Req.Request.t()} | {:error, term()}
  def prepare(%RequestPlan{transport: :req} = plan, input) do
    Adapter.prepare_request(
      plan.provider_module,
      plan.operation,
      plan.model,
      input,
      plan.options
    )
  end

  @doc """
  Sends an already prepared Req request using the supplied deadline.

  The operation owns when its deadline starts and any changes made after provider
  preparation. Every completed response, including non-2xx responses, reaches the
  result policy outside the request budget. Transport errors pass through without
  invoking the policy. A prepared request can be reused for subsequent executions.
  """
  @spec execute_prepared(
          Req.Request.t(),
          TimeoutBudget.deadline(),
          (Req.Response.t() -> {:ok, term()} | {:error, term()})
        ) :: {:ok, term()} | {:error, term()}
  def execute_prepared(%Req.Request{} = request, deadline, result_policy)
      when is_function(result_policy, 1) do
    with {:ok, response} <- TimeoutBudget.request(request, deadline) do
      result_policy.(response)
    end
  end

  @doc """
  Starts the selected v1 stream transport on its existing StreamServer.

  The operation boundary supplies an already resolved streaming plan and context.
  StreamServer still starts and monitors the transport task and owns backpressure,
  decoding, cancellation, and timeouts. Clients attach through the v1 provider
  adapter, including WebSocket HTTP fallback. Surface validation stays inside the
  provider callback; startup never dispatches on the plan's surface or API module.

  The caller owns response assembly, telemetry installation, fixture context, and
  cleanup if startup fails. Transport-specific errors retain their v1 wrappers.
  """
  @spec start_stream(RequestPlan.t(), ReqLLM.Context.t(), pid()) ::
          {:ok, pid(), ReqLLM.Streaming.Fixtures.HTTPContext.t() | nil, term()}
          | {:error, term()}
  def start_stream(%RequestPlan{} = plan, context, server_pid) do
    opts =
      plan.options
      |> Keyword.delete(:pricing_context)
      |> Keyword.put(:defer_http_events_until_telemetry?, true)

    start_stream_transport(plan, context, opts, server_pid)
  end

  defp start_stream_transport(
         %RequestPlan{transport: :websocket} = plan,
         context,
         opts,
         server_pid
       ) do
    case StreamServer.start_http(
           server_pid,
           plan.provider_module,
           plan.model,
           context,
           Keyword.put(opts, :stream_transport, :websocket),
           ReqLLM.Finch
         ) do
      {:ok, task_pid, http_context, canonical_json} ->
        {:ok, task_pid, http_context, canonical_json}

      {:error, reason} ->
        Logger.error("Failed to start WebSocket streaming: #{inspect(reason)}")
        {:error, {:websocket_streaming_failed, reason}}
    end
  end

  defp start_stream_transport(
         %RequestPlan{transport: :in_process} = plan,
         context,
         opts,
         server_pid
       ) do
    case StreamServer.start_in_process(
           server_pid,
           plan.provider_module,
           plan.model,
           context,
           opts
         ) do
      {:ok, task_pid, stream_context, canonical_request} ->
        {:ok, task_pid, stream_context, canonical_request}

      {:error, reason} ->
        Logger.error("Failed to start in-process streaming: #{inspect(reason)}")
        {:error, {:in_process_streaming_failed, reason}}
    end
  end

  defp start_stream_transport(%RequestPlan{transport: :finch} = plan, context, opts, server_pid) do
    case StreamServer.start_http(
           server_pid,
           plan.provider_module,
           plan.model,
           context,
           opts,
           Keyword.get(opts, :finch_name, ReqLLM.Finch)
         ) do
      {:ok, task_pid, http_context, canonical_json} ->
        {:ok, task_pid, http_context, canonical_json}

      {:error, {:provider_build_failed, {:http2_body_too_large, body_size, protocols}}} ->
        message = format_http2_error_message(body_size, protocols)
        Logger.error(message)
        {:error, {:http2_body_too_large, message}}

      {:error, reason} ->
        Logger.error("Failed to start HTTP streaming: #{inspect(reason)}")
        {:error, {:http_streaming_failed, reason}}
    end
  end

  defp format_http2_error_message(body_size, protocols) do
    size_kb = div(body_size, 1024)

    """
    Request body (#{size_kb}KB) exceeds safe limit for mixed HTTP/1+HTTP/2 Finch pools (64KB).

    This is due to a known issue in Finch's mixed-protocol ALPN implementation:
    https://github.com/sneako/finch/issues/265

    Your current pool configuration uses: #{inspect(protocols)}

    To fix this, configure ReqLLM to use HTTP/1-only pools (recommended default):

        config :req_llm,
          stream_pool_protocols: [:http1]

    If you know all target providers support HTTP/2, HTTP/2-only pools are also supported:

        config :req_llm,
          stream_pool_protocols: [:http2]

    See the ReqLLM README section "HTTP/2 Configuration (Advanced)" for more details:
    https://github.com/agentjido/req_llm#http2-configuration-advanced
    """
  end
end
