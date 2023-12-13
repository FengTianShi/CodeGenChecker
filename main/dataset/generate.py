import os
from openai import OpenAI

current_directory = os.path.dirname(__file__)

api_path = os.path.join(current_directory, "resource", "api-key.txt")
with open(api_path, "r") as f:
    key = f.read().strip()

requirement_path = os.path.join(current_directory, "resource", "requirement.txt")
with open(requirement_path, "r") as f:
    requirement = f.read().strip()

problem_path = "C://Users/Mining-Base/Downloads/Project_CodeNet/Project_CodeNet/problem_descriptions"
problem_list = os.listdir(problem_path)

client = OpenAI(api_key=key)
start = 2650
max_gen_num = 9999

for problem_name in problem_list:
    if start > 0:
        start -= 1
        continue

    max_gen_num -= 1
    if max_gen_num < 0:
        break

    with open(os.path.join(problem_path, problem_name), "r", encoding="utf-8") as f:
        problem_description = f.read().strip()

    problem_id = problem_name.split(".")[0]
    print(problem_id)

    # if bad request accured, skip problem
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": requirement,
                },
                {
                    "role": "user",
                    "content": problem_description,
                },
            ],
        )
    except:
        print("bad request, skip : " + problem_id)
        continue

    with open(
        os.path.join(current_directory, "generated", problem_id + ".java"), "w"
    ) as f:
        response = completion.choices[0].message.content

        x = "```java"
        y = "```"
        if x in response:
            response = response.split(x, 1)[1]
        if y in response:
            response = response.split(y, 1)[0]

        response = response.strip()
        f.write(response)
