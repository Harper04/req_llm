defmodule ReqLLM.Test.ProviderRegistration do
  @moduledoc """
  Temporary provider registration for synchronous tests.

  Restores the previous module for the provider ID on exit, or removes the
  registration if that ID was absent. Use in a test or per-test setup with
  a provider module exposing `provider_id/0`, such as those using `ReqLLM.Provider`.
  """

  alias ReqLLM.Providers

  @doc "Registers a provider and restores its prior registration when the test exits."
  @spec register_provider(module()) :: :ok
  def register_provider(module) do
    provider_id = module.provider_id()
    previous = Providers.get(provider_id)

    ExUnit.Callbacks.on_exit(fn ->
      case previous do
        {:ok, previous_module} ->
          ^provider_id = Providers.register!(previous_module)

        {:error, _error} ->
          Providers.unregister(provider_id)
      end
    end)

    ^provider_id = Providers.register!(module)
    :ok
  end
end
