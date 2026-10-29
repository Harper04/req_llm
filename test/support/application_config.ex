defmodule ReqLLM.TestSupport.ApplicationConfig do
  @moduledoc """
  Temporary ReqLLM application configuration for synchronous ExUnit tests.

  Snapshots each configured key before setting it and restores its original
  value with `ExUnit.Callbacks.on_exit/1`. Keys that were absent are deleted;
  explicitly configured `nil` values remain present. Use only in tests or setup
  callbacks of cases with `async: false`, since application configuration is global.
  """

  @doc """
  Sets ReqLLM configuration until the current test exits, even if it fails.

      ReqLLM.TestSupport.ApplicationConfig.put_env(
        schema_cache_max_bytes: 1_024,
        schema_cache_eviction_strategy: ReqLLM.Schema.EvictionStrategy.FIFO
      )

  Multiple calls restore their snapshots in reverse order on exit.
  """
  @spec put_env(keyword()) :: :ok
  def put_env(config) do
    previous =
      Enum.map(config, fn {key, _value} -> {key, Application.fetch_env(:req_llm, key)} end)

    ExUnit.Callbacks.on_exit(fn ->
      Enum.each(previous, fn
        {key, {:ok, value}} -> Application.put_env(:req_llm, key, value)
        {key, :error} -> Application.delete_env(:req_llm, key)
      end)
    end)

    Enum.each(config, fn {key, value} -> Application.put_env(:req_llm, key, value) end)
  end
end
