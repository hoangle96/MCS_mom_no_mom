tasks = ['MPS', 'MPM', 'NMS', 'NMM']


bricks_cols = ['bricks.ordinal', 'bricks.onset', 'bricks.offset',
               'bricks.carry', 'bricks.roll_throw', 'bricks.knock', 'bricks.build']
pig_cols = ['pig.ordinal', 'pig.onset',
            'pig.offset', 'pig.carry', 'pig.roll_throw']
popuppals_cols = ['popuppals.ordinal', 'popuppals.onset',
                  'popuppals.offset', 'popuppals.carry', 'popuppals.push_buttons']
xylophone_cols = ['xylophone.ordinal', 'xylophone.onset',
                  'xylophone.offset', 'xylophone.carry', 'xylophone.push_buttons']
shape_sorter_cols = ['shape_sorter.ordinal', 'shape_sorter.onset', 'shape_sorter.offset',
                     'shape_sorter.carry', 'shape_sorter.roll_throw', 'shape_sorter.empty']
shape_sorter_blocks_cols = ['shape_sorter_blocks.ordinal', 'shape_sorter_blocks.onset', 'shape_sorter_blocks.offset',
                            'shape_sorter_blocks.carry', 'shape_sorter_blocks.roll_throw', 'shape_sorter_blocks.fit']
broom_cols = ['broom.ordinal', 'broom.onset',
              'broom.offset', 'broom.carry', 'broom.whack']
clear_ball_cols = ['clear_ball.ordinal', 'clear_ball.onset', 'clear_ball.offset',
                   'clear_ball.carry', 'clear_ball.roll_throw', 'clear_ball.kick']
balls_cols = ['balls.ordinal', 'balls.onset', 'balls.offset',
              'balls.carry', 'balls.roll_throw', 'balls.kick']
food_cols = ['food.ordinal', 'food.onset', 'food.offset',
             'food.carry', 'food.roll_throw', 'food.eat', 'food.feed']
grocery_cart_cols = ['grocery_cart.ordinal', 'grocery_cart.onset', 'grocery_cart.offset',
                     'grocery_cart.carry', 'grocery_cart.push', 'grocery_cart.fill_empty']
stroller_cols = ['stroller.ordinal', 'stroller.onset', 'stroller.offset', 'stroller.carry',
                 'stroller.push', 'stroller.sit', 'stroller.climb', 'stroller.fill_empty']

toys_list = ['bricks', 'pig', 'popuppals', 'xylophone', 'shape_sorter', 'shape_sorter_blocks',
             'broom', 'clear_ball', 'balls', 'food', 'grocery_cart', 'stroller', 'bucket']
print(len(toys_list))

stationary_toys_list = ['shape_sorter', 'shape_sorter_blocks','xylophone','bricks', 'pig', 'popuppals']
mobile_toys_list = ['grocery_cart', 'food', 'bucket','balls','stroller','broom', 'clear_ball']

# map task to the correct dictionary
toy_to_task_dict = {'MPS': stationary_toys_list, 'MPM': mobile_toys_list,
                    'NMS': stationary_toys_list, 'NMM': mobile_toys_list}
# map toy to the columns
toys_dict = {'bricks': bricks_cols, 'pig': pig_cols, 'popuppals': popuppals_cols, 'xylophone': xylophone_cols, 'shape_sorter': shape_sorter_cols,
             'shape_sorter_blocks': shape_sorter_blocks_cols, 'broom': broom_cols, 'clear_ball': clear_ball_cols, 'balls': balls_cols,
             'food': food_cols, 'grocery_cart': grocery_cart_cols, 'stroller': stroller_cols }

toys_of_interest_dict = {'MPS': {'shape_sorter': 'shape_sorter_blocks'}, 'NMS': {'shape_sorter': 'shape_sorter_blocks'}, 'NMM':{'grocery_cart': ['food', 'balls'], 'bucket': ['food', 'balls']},  'MPM':{'grocery_cart': ['food', 'balls'], 'bucket': ['food', 'balls']}}
non_compatible_toys_dict = {'bricks': ['pig', 'popuppals','xylophone', 'shape_sorter', 'shape_sorter_blocks'],\
                            'pig': ['bricks', 'popuppals','xylophone', 'shape_sorter', 'shape_sorter_blocks'],\
                            'popuppals': ['bricks', 'pig', 'xylophone', 'shape_sorter', 'shape_sorter_blocks'],\
                            'xylophone': ['bricks', 'pig', 'popuppals', 'shape_sorter', 'shape_sorter_blocks'],\
                            'shape_sorter': ['bricks', 'pig', 'popuppals', 'xylophone'],\
                            'shape_sorter_blocks': ['bricks', 'pig', 'popuppals', 'xylophone'], \
                            'broom': ['bucket', 'stroller', 'grocery_cart', 'balls', 'food'],\
                            'bucket': ['broom', 'stroller', 'clear_ball', 'grocery_cart'],\
                            'clear_ball': ['broom', 'bucket', 'stroller', 'grocery_cart', 'balls', 'food'],\
                            'balls': ['broom', 'stroller', 'clear_ball'],\
                            'food': ['broom', 'stroller', 'clear_ball'],\
                            'grocery_cart': ['broom', 'stroller', 'clear_ball', 'bucket'],
                            'stroller': ['broom', 'bucket', 'grocery_cart', 'balls', 'food', 'clear_ball']}
# threshold, in ms to consider toy switching
small_no_ops_threshold_dict = {'MPS': 7000, 'MPM': 7000, 'NMS': 7000, 'NMM': 7000}
condition_name = {"MPS" : "With caregiver, fine motor toys",\
                "NMS" : "Without caregiver, fine motor toys",\
                "NMM" : "Without caregiver, gross motor toys",\
                "MPM" : "With caregiver, gross motor toys"
                }
state_color_dict = {"0":'dimgrey',  "1":'green', "2":'purple', "3":'darkorange', "4":'darkslateblue',  "5":'crimson', "6":'darkolivegreen', "7":'blue'}
toy_colors_dict = {'bricks': 'blue', 'pig': 'purple', 'popuppals': 'green', 'xylophone': 'red', 'shape_sorter': 'skyblue',
                                    'shape_sorter_blocks': 'salmon', 'broom': 'purple', 'clear_ball': 'teal', 'balls': 'cadetblue',
                                    'food': 'chocolate', 'grocery_cart': 'dodgerblue', 'stroller': 'violet', 'bucket': 'navy', 'no_toy': "slategrey"}

state_color_dict_shades = {"0":'lightgrey',  "1":'red', "2":'salmon', "3":'royalblue', "4":'darkblue',  "5":'midnightblue', "6":'midnightblue', "7":'blue'}

# state_name_dict = {
#     5: {
#         1:{
#             3: {2: "no_toy", 1: 'F', 0: 'E'},
#             4: {3: "no_toy", 0: "F+", 1: 'E',  2: 'E+'},
#             5: {4: "no_toy", 0: 'F+', 2: "F", 1:"E", 3: 'E+'},
#             6: {5: "no_toy", 2: "F+", 3: 'F', 0: 'E', 4: "E+", 1: 'E++'}
#         },
#         1.5:{
#             3: {2: "no_toy", 1: 'F', 0: 'E'},
#             4: {3: "no_toy", 0: "F", 2: 'E',  1: 'E+'},
#             5: {4: "no_toy", 2: 'F+', 3: "F", 0:"E", 1: 'E+'},
#             6: {5: "no_toy", 2: "F+", 3: 'F', 4: 'E', 0: "E+", 1: 'E++'}
#         },
#         2:{
#             3: {2: "no_toy", 1: 'F', 0: 'E'},
#             4: {3: "no_toy", 1: "F", 2: 'E',  0: 'E+'},
#             5: {4: "no_toy", 3: 'F+', 0: "F", 2:"E", 1: 'E+'},
#             6: {5: "no_toy", 1: "F+", 0: 'F', 2: 'E', 4: "E+", 3: 'E++'}
#         }
#     },
#     7: {
#         1:{
#             3: {2: "no_toy", 0: 'F', 1: 'E'},
#             4: {3: "no_toy", 2: "F", 1: 'E',  0: 'E+'},
#             5: {4: "no_toy", 2: 'F+', 1: "F", 0:"E", 3: 'E+'},
#             6: {5: "no_toy", 2: "F+", 4: 'F', 3: 'E', 0: "E+", 1: 'E++'}
#         },
#         1.5:{
#             3: {2: "no_toy", 1: 'F', 0: 'E'},
#             4: {3: "no_toy", 0: "F", 1: 'E', 2: 'E+'},
#             5: {4: "no_toy", 2: 'F+', 1: "F", 0:"E", 3: 'E+'},
#             6: {5: "no_toy", 0: "F+", 3: 'F+', 2: 'F', 4: "E", 1: 'E+'}
#         },
#         2:{
#             3: {2: "no_toy", 0: 'F', 1: 'E'},
#             4: {3: "no_toy", 0: "F", 2: 'E',  1: 'E+'},
#             5: {4: "no_toy", 1: 'F+', 0: "F", 3:"E", 2: 'E+'},
#             6: {5: "no_toy", 3: 'F+', 0: "F", 2:"E", 4:"E+", 1: 'E++'},
#         }
#     },
#     10:{
#         1:{
#             3: {2: "no_toy", 0: 'F', 1: 'E'},
#             4: {3: "no_toy", 0: "F", 1: 'E',  2: 'E+'},
#             5: {4: "no_toy", 0: 'F+', 1: "F", 3:"E", 2: 'E+'},
#             6: {5: "no_toy", 0: 'F+', 1: "F", 2:"E", 4:"E+", 3: 'E++'},
#         },
#         1.5:{
#             3: {2: "no_toy", 1: 'F', 0: 'E'},
#             4: {3: "no_toy", 1: 'F', 0: 'E', 2: 'E+'},
#             5: {4: "no_toy", 2: 'F+', 3: "F", 1:"E", 0: 'E+'},
#             6: {5: "no_toy", 2: 'F+', 4: "F", 0:"E", 1:"E+", 3: 'E++'},
#         },
#         2:{
#             3: {2: "no_toy", 0: 'F', 1: 'E'},
#             4: {3: "no_toy", 2: 'F', 1: 'E', 0: 'E+'},
#             5: {4: "no_toy", 1: 'F+', 3: "F", 0:"E", 2: 'E+'},
#             6: {5: "no_toy", 1: 'F+', 3: "F", 0:"E", 2:"E+", 4: 'E++'},
#         }
#     }
# }