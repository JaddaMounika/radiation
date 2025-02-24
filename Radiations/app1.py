import pickle
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder

model = pickle.load(open('model.pkl', 'rb'))

# Create label encoders for categorical variables
gadget_encoder = LabelEncoder()
company_encoder = LabelEncoder()

# Assuming 'Laptop', 'Desktop', 'Tablet' are the possible values for 'gadget'
# and 'Apple', 'Microsoft', 'Lenovo' are the possible values for 'company'
possible_gadgets = ['Laptop', 'Smartwatch', 'Smartphone','Headphones']
possible_companies = ['Solutions Group', 'Enterpri,Innovations Inc','Tech Groupses Group', 'Enterprises Ltd','Solutions Corp','Tech Inc','Tech crop','system crop','Gadget Group','Tech Ltd','Gadget Ltd']

# Fit the label encoders
gadget_encoder.fit(possible_gadgets)
company_encoder.fit(possible_companies)

app = Flask(__name__)

@app.route('/')
def htmlPage():
    return render_template('list1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collecting data from the form
        gadget = request.form['gadget']
        company = request.form['company']
        warranty = float(request.form['warranty'])
        manufacturing_year = float(request.form['manufacturing_year'])
        expiry_year = float(request.form['expiry_year'])
        
        # Transforming categorical variables to numerical values
        gadget_encoded = gadget_encoder.transform([gadget])[0]
        company_encoded = company_encoder.transform([company])[0]
        
        # Making the prediction
        result = model.predict([[gadget_encoded, company_encoded, warranty, manufacturing_year, expiry_year]])
        print(result)
        
        # Returning the prediction result as a response
        return render_template('list1.html', prediction=result[0])

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5001, debug=True)
