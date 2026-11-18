import requests

BASE_URL = "http://localhost:8080"   # replace with your cloud URL after deploying

passenger = {"Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0,
             "Parch": 0, "Fare": 80, "Embarked": "C"}

print(requests.get(f"{BASE_URL}/health").json())
print(requests.post(f"{BASE_URL}/predict", json=passenger).json())
