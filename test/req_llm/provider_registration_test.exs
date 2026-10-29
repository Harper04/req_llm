defmodule ReqLLM.Test.ProviderRegistrationTest do
  use ExUnit.Case, async: false

  test "restores a different module previously registered under the same ID" do
    assert_cleanup(:registered, :success)
  end

  test "removes the registration when the ID was previously absent" do
    assert_cleanup(:unregistered, :success)
  end

  test "restores the prior registration state after a failing test" do
    assert_cleanup(:registered, :failure)
    assert_cleanup(:unregistered, :failure)
  end

  defp assert_cleanup(initial_state, outcome) do
    initial_registration =
      case initial_state do
        :registered -> "ReqLLM.Providers.register!(OriginalProvider)"
        :unregistered -> "ReqLLM.Providers.unregister(:registration_helper_test)"
      end

    expected_registration =
      case initial_state do
        :registered -> "{:ok, OriginalProvider}"
        :unregistered -> "{:error, _}"
      end

    test_result = if outcome == :failure, do: ~s|flunk("intentional failure")|, else: ":ok"
    expected_failures = if outcome == :failure, do: 1, else: 0

    script = """
    ExUnit.start(autorun: false)
    Process.register(self(), :registration_observer)

    defmodule OriginalProvider do
      use ReqLLM.Provider,
        id: :registration_helper_test,
        default_base_url: "https://original.example"
    end

    defmodule TemporaryProvider do
      use ReqLLM.Provider,
        id: :registration_helper_test,
        default_base_url: "https://temporary.example"
    end

    #{initial_registration}

    defmodule RegistrationCase do
      use ExUnit.Case, async: false

      test "temporary registration" do
        assert :ok = ReqLLM.Test.ProviderRegistration.register_provider(TemporaryProvider)
        assert {:ok, TemporaryProvider} = ReqLLM.Providers.get(:registration_helper_test)
        send(:registration_observer, :temporary_registered)
        #{test_result}
      end
    end

    %{total: 1, failures: #{expected_failures}} = ExUnit.run()
    receive do
      :temporary_registered -> :ok
    after
      0 -> raise "test failed before registering the temporary provider"
    end
    #{expected_registration} = ReqLLM.Providers.get(:registration_helper_test)
    """

    code_paths = Enum.flat_map(:code.get_path(), &["-pa", List.to_string(&1)])

    {output, status} =
      System.cmd(System.find_executable("elixir"), code_paths ++ ["-e", script],
        stderr_to_stdout: true
      )

    assert status == 0, output
  end
end
