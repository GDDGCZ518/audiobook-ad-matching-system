import json
import requests

def request_claude_api(prompt):
    """调用大模型API进行翻译"""
   
    model = "gpt-4o"
    
    url = f"http://deepgate.ximalaya.local/{model}/api/v1/chat/completions"
    headers = {
        "Authorization": "Bearer 161332424d5d43649599351e2a20f0f1",
        "Content-Type": "application/json"
    }
    data = {
        "max_tokens": 8092,
        "model": f"{model}",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        res = response.json()["choices"][0]["message"]["content"]
    else:
        print(f"请求失败，状态码：{response.status_code}")
        print(response.text)
        raise Exception(f"API请求失败: {response.status_code}")
    
    return res
