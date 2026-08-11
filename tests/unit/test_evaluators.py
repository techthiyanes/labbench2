from unittest.mock import AsyncMock, MagicMock

import pytest

from evals.evaluators import HybridEvaluator, RewardFunctionEvaluator, extract_answer


class TestExtractAnswer:
    def test_simple(self):
        output = "The answer is <answer>42.5</answer>"
        regex = r"(?P<answer>[\d.]+)"
        assert extract_answer(output, regex) == {"answer": "42.5"}

    def test_multiple_groups(self):
        output = "<answer>ATCG, GCTA</answer>"
        regex = r"(?P<forward>\w+),\s*(?P<reverse>\w+)"
        assert extract_answer(output, regex) == {"forward": "ATCG", "reverse": "GCTA"}

    def test_no_match(self):
        output = "No tags here"
        regex = r"(?P<answer>\d+)"
        assert extract_answer(output, regex) is None

    def test_no_regex(self):
        output = "<answer>42</answer>"
        assert extract_answer(output, None) is None


class TestHybridEvaluatorRouting:
    @pytest.fixture
    def evaluator(self):
        return HybridEvaluator()

    @pytest.mark.asyncio
    async def test_routes_seqqa2_to_reward(self, evaluator):
        evaluator.reward_evaluator.evaluate = AsyncMock(return_value=0.8)
        evaluator.llm_evaluator.evaluate = AsyncMock(return_value=1.0)

        ctx = MagicMock(metadata={"tag": "seqqa2"})
        await evaluator.evaluate(ctx)

        evaluator.reward_evaluator.evaluate.assert_called_once()
        evaluator.llm_evaluator.evaluate.assert_not_called()

    @pytest.mark.asyncio
    async def test_routes_cloning_to_reward(self, evaluator):
        evaluator.reward_evaluator.evaluate = AsyncMock(return_value=0.8)
        evaluator.llm_evaluator.evaluate = AsyncMock(return_value=1.0)

        ctx = MagicMock(metadata={"tag": "cloning"})
        await evaluator.evaluate(ctx)

        evaluator.reward_evaluator.evaluate.assert_called_once()
        evaluator.llm_evaluator.evaluate.assert_not_called()

    @pytest.mark.asyncio
    async def test_routes_litqa3_to_llm(self, evaluator):
        evaluator.reward_evaluator.evaluate = AsyncMock(return_value=0.8)
        evaluator.llm_evaluator.evaluate = AsyncMock(return_value=1.0)

        ctx = MagicMock(metadata={"tag": "litqa3"})
        await evaluator.evaluate(ctx)

        evaluator.llm_evaluator.evaluate.assert_called_once()
        evaluator.reward_evaluator.evaluate.assert_not_called()


@pytest.mark.asyncio
async def test_cloning_evaluator_accepts_decoded_validator_params(monkeypatch, tmp_path):
    validator_params = {"enzyme_1": "BsaI", "edit_distance_threshold": 0.05}
    reference_path = tmp_path / "question-id_assembled.fa"
    reward = AsyncMock(return_value=(1.0, "ok"))
    resolver = MagicMock(return_value=reference_path)
    monkeypatch.setattr("evals.evaluators.cloning_reward", reward)
    monkeypatch.setattr("evals.evaluators.resolve_file_path", resolver)
    ctx = MagicMock(
        metadata={
            "tag": "cloning",
            "id": "question-id",
            "validator_params": validator_params,
        },
        inputs={"files_path": str(tmp_path)},
        output="protocol",
    )

    result = await RewardFunctionEvaluator().evaluate(ctx)

    resolver.assert_called_once_with("question-id_assembled.fa", None)
    reward.assert_awaited_once_with(
        answer="protocol",
        base_dir=tmp_path,
        reference_path=reference_path,
        validator_params=validator_params,
    )
    assert result.value == 1.0
    assert result.reason == "ok"
