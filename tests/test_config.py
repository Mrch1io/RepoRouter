#测试配置文件
#通过src.调用文件
from src.config import settings


def test_models_are_not_retired():
    #deepseek-chat 已经在 2026-07 下线
    assert settings.model_name != "deepseek-chat"
    assert settings.routing_backend in ("native","phase2a")


#直接运行时打印一下当前配置（不打印key）
if __name__ == "__main__":
    print("API key 已填写:" , bool(settings.deepseek_api_key))
    print("模型名:",f"{settings.model_name}")
