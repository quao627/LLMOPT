classification_system_info = "You are an expert in the field of operations and optimization. "



def classification_type(ques):
    ans = f"""Optimization problems can usually be classified into one of the following categories: 
    ```
    Linear Programming
    Integer Programming
    Mixed Integer Programming
    Nonlinear Programming
    Multi-objective Programming
    Quadratic Programming
    Dynamic Programming
    Stochastic Programming
    Combinatorial Optimization
    Others
    ```
    The following is an optimization problem. Please determine which category it belongs to: 
    ```
    {ques}
    ```
    You need to answer in the following format: 
    ```
    The category of this problem is: [One of the categories mentioned above]
    ```
    """
    return ans


def classification_background(ques):
    ans = f"""Optimization problems can usually be classified into one of the following categories: 
    ```
    Agriculture
    Energy
    Health
    Retail
    Environment
    Education
    Financial Services
    Transportation
    Public Utilities
    Manufacturing
    Software
    Construction
    Legal
    Customer Service
    Entertainment
    Others
    ```
    The following is an optimization problem. Please determine which category it belongs to: 
    ```
    {ques}
    ```
    You need to answer in the following format: 
    ```
    The category of this problem is: [One of the categories mentioned above]
    ```
    """
    return ans

