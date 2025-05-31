from pydantic import BaseModel
from typing import List


class CreditInput(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    BILL_AMT1: float
    PAY_AMT1: float
    PAY_AMT2: float
