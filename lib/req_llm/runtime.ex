defmodule ReqLLM.Runtime do
  @moduledoc false

  alias ReqLLM.Provider.Adapter
  alias ReqLLM.RequestPlan
  alias ReqLLM.TimeoutBudget

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
    with {:ok, request} <-
           Adapter.prepare_request(
             plan.provider_module,
             plan.operation,
             plan.model,
             input,
             plan.options
           ),
         {:ok, response} <- TimeoutBudget.request(request, deadline) do
      result_policy.(response)
    end
  end
end
