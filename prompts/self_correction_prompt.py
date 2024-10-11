def self_correction(ques, five, code, output, error):
    return f"""For the following optimization problem, modeling is performed, and pyomo code is generated and executed based on the modeling. Please judge whether the modeling and code are correct.
        The problem is as follows.

        {ques}

        The five-element formulation is as follows.

        {five}

        The code is as follows.

        {code}

        Run the code and get the following running information. 

        {output}
        {error}

        Please judge whether the above five-element and code are correct, and give your analysis according to the template below.

        ```
        The five-element is [Fill in True/False here].

        The code is [Fill in True/False here].

        Analysis:
        [Fill in your analysis here]
        ```"""
