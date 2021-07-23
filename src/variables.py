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

stationary_toys_list = ['bricks', 'pig', 'popuppals',
                        'xylophone', 'shape_sorter', 'shape_sorter_blocks']
mobile_toys_list = ['broom', 'bucket', 'stroller',
                    'grocery_cart', 'balls', 'food']

# map task to the correct dictionary
toy_to_task_dict = {'MPS': stationary_toys_list, 'MPM': mobile_toys_list,
                    'NMS': stationary_toys_list, 'NMM': mobile_toys_list}
# map toy to the columns
toys_dict = {'bricks': bricks_cols, 'pig': pig_cols, 'popuppals': popuppals_cols, 'xylophone': xylophone_cols, 'shape_sorter': shape_sorter_cols,
             'shape_sorter_blocks': shape_sorter_blocks_cols, 'broom': broom_cols, 'clear_ball': clear_ball_cols, 'balls': balls_cols,
             'food': food_cols, 'grocery_cart': grocery_cart_cols, 'stroller': stroller_cols}

toys_of_interest_dict = {'MPS': {'shape_sorter': 'shape_sorter_blocks'}, 'NMS': {'shape_sorter': 'shape_sorter_blocks'}, 'NMM':{'grocery_cart': ['food', 'balls'], 'bucket': ['food', 'balls']},  'MPM':{'grocery_cart': ['food', 'balls'], 'bucket': ['food', 'balls']}}
