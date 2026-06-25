from pydantic import BaseModel

class HouseInput(BaseModel):
    OverallQual : int
    GrLivArea : int
    GarageCars : int 
    TotalBsmtSF : int
    YearBuilt : int
    
