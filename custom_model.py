import numpy as np
import pandas as pd
import math

from sklearn.base import BaseEstimator, ClassifierMixin

class my_GB_3(BaseEstimator, ClassifierMixin):
    """
    Custom Gaussian Naive Bayes classifier, can handle NaN values in training dataset as well as in predict queries.
    
    """
    
    def __init__(self):
        self.dfs = {}
        self.prior_probabilities = {}
        self.std_mean_dict= {}
        self.feature_count = -1
        
    def shape_checker(self, input_array):
        
        it = iter(input_array)
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
             raise ValueError('All list items must contain the same number of features')
    
    def fit(self, Xt, yt):
        
        
        def nan_std_mean_calculator():
            
            #have std and mean for nan cleaned transposed arrays
            for key, value in self.dfs.items():
                self.std_mean_dict[key] = {}

                for column in self.dfs[key]:
                    
                    numpystd = np.nanstd(self.dfs[key][column].values)
                    
                    numpymean = np.nanmean(self.dfs[key][column].values)
                    
                    self.std_mean_dict[key][column] = {"std" : numpystd, "mean": numpymean}

                    
        
   
        def prior_probability_generator(target_array):
            """
                Generates dictionary keys based on target values and values being the prior probabilities of of target values
                return value is said dictionary
            """
            return dict(zip(target_array.value_counts(normalize=True).index,target_array.value_counts(normalize=True).values))

        def data_splitter(full_data, target_feature):
            """
            Populates a dictionary with key values based on target feature values
            Returns said dictionary

            """

            dfs = {}

            for value in full_data[target_feature].value_counts().index.tolist():
                    dfs[value] = full_data[full_data[target_feature]== value].drop(columns=[target_feature])

            return dfs
        
        
        
        #checking the shape of the input is correct
        self.shape_checker(Xt)
        
        #Xt and Yt simply represent the feature array and target features array respectively
        self.Xt = Xt
        self.yt = yt
        
        df = pd.DataFrame(Xt)
        
        #limiting the nan values of a feature to 45% as model sees a degredation in performance beyond this threshold        
        if not all(i< 0.45 for i in df.isna().mean().to_list()):
            raise ValueError('Excessive proportion of NaN values detected. Percentage missing must be below 45%.')
            
            
        df['y'] = yt
        
        self.dfs = data_splitter(df, 'y')
        self.prior_probabilities= prior_probability_generator(df['y'])
        self.feature_count = len(Xt[0])
        
        nan_std_mean_calculator()

        
    def nan_std_mean_calculator(self):
            
        #have std and mean for nan cleaned transposed arrays
        for key, value in self.dfs.items():
            self.std_mean_dict[key] = {}

            for column in self.dfs[key]:
                values = self.dfs[key][column].values
                values = values[~np.isnan(values)]
                std = np.std(values)
                mean = np.mean(values)
                self.std_mean_dict[key][column] = {"std" : std, "mean": mean}   
                
        print(self.std_mean_dict)
        
    def predict(self, X_array):
        
        def array_input_length_checker(input_array):
            if len(X_array[0]) != self.feature_count:
                raise ValueError('Predicted item length must be the same as fit training set')
                
        
        #checking that input has same number of features as fit was trained on
        array_input_length_checker(X_array)
        
        #checking that predict input array is all of same length
        self.shape_checker(X_array)
        
         
        
        output_array = []
        
        for element in X_array:
            output_array.append(self.predict_single(element))
            
        return output_array
        

    def predict_single(self, input_value):
        
        
        def not_nan_indexes(input_list):
            """
                Returns a list of indices where values are not 
            """
            not_nan = np.argwhere(~np.isnan(input_list)).tolist()
            return [item for sublist in not_nan for item in sublist]
        
        def feature_conditional_probability_generator(x_value, feature, class_name):
            """
                Calculates the conditional probability of a feature value based on the associated x feature value passed
                Formula was sourced from assignement problem statement as requested
                return value is float of the operation result

            """
            
            exponential_component = np.exp(-((x_value - self.std_mean_dict[class_name][feature]['mean']) ** 2 / (2 * self.std_mean_dict[class_name][feature]['std'] ** 2)))
            return (1 / (np.sqrt(2 * np.pi) *  self.std_mean_dict[class_name][feature]['std']))*exponential_component
        
        
        def single_class_conditionals_generator(x_value, class_name, non_nan_list):
            """
                Generates a dictionary of the all conditional values for a single class and assignes them to a dictionary
                Probabilities for each feature of the class are calculated using the feature_conditional_probability_generator() function 
                returns said dictionary

            """

            class_conditionals = {}
            
            #here instead of iterating over all columns in the dataframe, i can simple use a for element in non_nan list
            #this will allow me to only calculate the conditionals on the non nan values present
            
            for element in non_nan_list:
                class_conditionals[element] = feature_conditional_probability_generator(x_value[element], element, class_name)

            return class_conditionals
        
        def all_class_conditional_dict_generator(x_value, non_nan_list):
            """
                Generates a dictionary of all class probability values and assigns them to a dictionary
                Probabilties for each class conditionals are calculated using the single_class_conditionals_generator() fucntion
                return value is said dictionary

            """

            all_class_conditionals = {}

            for class_name in self.dfs.keys():
                all_class_conditionals[class_name] = single_class_conditionals_generator(x_value, class_name, non_nan_features)

            return all_class_conditionals
        
        def class_probability_generator(class_name):
            """
            Generates the probability of each target class for the given predict input
            Class prior probabilities are sourced from the prior_probabilities dictionary generated from the fit() method
            Class probabilities are generated via Naive baysienne by multiplying the class prior probability by the product of all class conditionals
            return value is the class probability float value

            """
            
             
            class_prior_prob = self.prior_probabilities[class_name]
            
            #as i designed my all_class_conditional generator to only create non-nan conditionals this remains constant
            class_probability = class_prior_prob * (np.prod(list(all_class_conditionals[class_name].values())))

            return class_probability
        
        def all_class_probability_generator():
            """
            Generates a dictionary containing the class probabilties for each class of the target feature
            Class probabilties are generated from the class_probability_generator() function
            return value is said dictionary

            """

            class_probabilities = {}

            for class_name in self.dfs.keys():
                class_probabilities[class_name] = class_probability_generator(class_name)

            return class_probabilities
        
        def class_tagger(normalized_probability_dict):
            """
                Returns the key with max value from probability dict

            """
            return max(normalized_probability_dict, key=normalized_probability_dict.get)
        
        
        #input validation on predict value
        if (np.count_nonzero(np.isnan(input_value))/len(input_value) == 1.0):
            raise ValueError("Input must have at least one non-nan value")
            
        #retrieving the index of non-nan values
        non_nan_features = not_nan_indexes(input_value)
        
        #generating available conditional probability values for non-nan features
        all_class_conditionals = all_class_conditional_dict_generator(input_value, non_nan_features)
        
        #obtaining highest probability class
        predicted_probability = class_tagger(all_class_probability_generator())
        
        return predicted_probability   