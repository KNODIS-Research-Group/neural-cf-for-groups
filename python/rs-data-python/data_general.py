"""
Organized data abstraction layer with oo and mixins
"""

"""
    This class needs:
        - x_train
        - x_val
        - y_train
        - y_val

        - x_test
        - y_test

        - num_users
        - num_items

        - rating_min
        - rating_max
"""
class GeneralRS():
    def get_train_val(self):
        if hasattr(self, 'x_val'):
            return (self.x_train, self.x_val, self.y_train, self.y_val)
        else:
            return (self.x_train, [], self.y_train, [])

    def get_test(self):
        return (self.x_test, self.y_test)

    def get_shape(self):
        return (2,) # user_id and item_id
    
    def get_shape_splited(self):
        return (1,) # call it for user and call it for item
    
    def get_output_units(self):
        return 1
    
    def get_num_users(self):
        return self.num_users
    
    def get_num_items(self):
        return self.num_items

    def get_rating_max(self):
        return self.rating_max

    def get_rating_min(self):
        return self.rating_min    
    
    code = "tbd" # TBD in subclass
    def get_data_code(self):
        return self.code
    
    def info(self):
        print(type(self.get_num_items()))
        print(type(self.num_items))
        return  "DS: %s\n" \
                "Train rows: %i\n" \
                "Val rows: %i\n" \
                "Test rows: %i\n" \
                "Users: %i\n" \
                "Items: %i\n" \
                "Ratings: %i - %i\n" \
                % (
                    self.code,
                    len(self.x_train),
                    len(self.x_val) if hasattr(self, 'x_val') else 0,
                    len(self.x_test) if hasattr(self, 'x_test') else 0,
                    self.num_users,
                    self.num_items,
                    self.get_rating_min(),
                    self.get_rating_max()
                )
