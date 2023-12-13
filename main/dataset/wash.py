import os

current_directory = os.path.dirname(__file__)

generated_path = os.path.join(current_directory, "gen_test")
generated_list = os.listdir(generated_path)

problem_solution_path = (
    "C://Users/Mining-Base/Downloads/Project_CodeNet/Project_CodeNet/data"
)


for generated_name in generated_list:
    problem_id = generated_name.split(".")[0]
    # find solution by problem id in problem_solution_path
    solution_path = os.path.join(problem_solution_path, problem_id, "Java")
    if not os.path.exists(solution_path):
        continue

    solution_list = os.listdir(solution_path)
    # pick the first solution
    solution_name = solution_list[0]
    with open(os.path.join(solution_path, solution_name), "r", encoding="utf-8") as f:
        solution = f.read().strip()

    # write solution to human folder
    with open(
        os.path.join(current_directory, "hu_test", problem_id + ".java"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(solution)

# remove all comments
