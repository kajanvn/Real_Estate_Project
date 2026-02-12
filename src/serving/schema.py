from pydantic import BaseModel

class HouseFeatures(BaseModel):
    X1_transaction_date: float
    X2_house_age: float
    X3_distance_to_MRT: float
    X4_number_of_convenience_stores: int
    X5_latitude: float
    X6_longitude: float
