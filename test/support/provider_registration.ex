defmodule ReqLLM.Test.ProviderRegistration do
  @moduledoc """
  Temporary provider registration for synchronous tests.

  Restores the previous module for the given provider ID on exit, or removes
  the registration if that ID was absent. Use in a test or per-test setup;
  the provider ID must match the module's registration ID.
  """

  alias ReqLLM.Providers

  @doc "Registers a provider and restores its prior registration when the test exits."
  @spec register_provider(atom(), module()) :: :ok
  def register_provider(provider_id, module) do
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
