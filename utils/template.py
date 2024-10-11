from prompts.generate_prompt import *
from prompts.classification_prompt import *
import html


class Template:
    def __init__(self):
        self.system_info = "You are an AI assistant."
    
    def get_system_info(self):
        return self.system_info


class ClassificationTypeTemplate(Template):
    def __init__(self):
        super().__init__()
        self.system_info = classification_system_info

    def __call__(self, ques, encode=False):
        if encode:
            ques = html.escape(ques)
        return classification_type(ques)


class ClassificationBackgroundTemplate(Template):
    def __init__(self):
        super().__init__()
        self.system_info = classification_system_info

    def __call__(self, ques, encode=False):
        if encode:
            ques = html.escape(ques)
        return classification_background(ques)


class LabelingTemplate(Template):
    def __init__(self):
        super().__init__()
        self.system_info = generate_system_info

    def __call__(self, call_type, **kwag):
        encode = kwag['encode'] if 'encode' in kwag else False
        if call_type == 'q2f':
            return self.q2f(ques=kwag['ques'], encode=encode)
        elif call_type == 'q2c':
            return self.q2c(ques=kwag['ques'], encode=encode)
        elif call_type == 'qf2c':
            return self.qf2c(ques=kwag['ques'], five=kwag['five'], encode=encode)
        elif call_type == 'f2c':
            return self.f2c(five=kwag['five'], encode=encode)
        else:
            return None

    def q2f(self, ques, encode=False):
        if encode:
            ques = html.escape(ques)
        return Q2F(ques)

    def q2c(self, ques, encode=False):
        if encode:
            ques = html.escape(ques)
        return Q2C(ques)

    def qf2c(self, ques, five, encode=False):
        if encode:
            ques = html.escape(ques)
            five = html.escape(five)
        return QF2C(ques, five)

    def f2c(self, five, encode=False):
        if encode:
            five = html.escape(five)
        return F2C(five)
