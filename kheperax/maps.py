from kheperax.geoms import Segment, Pos

KHERPERAX_MAZES = {}

# # Standard Khaperax maze
# KHERPERAX_MAZES["standard"] = {
#     "segments": [
#         Segment(Pos(0.25, 0.25), Pos(0.25, 0.75)),
#         Segment(Pos(0.14, 0.45), Pos(0., 0.65)),
#         Segment(Pos(0.25, 0.75), Pos(0., 0.8)),
#         Segment(Pos(0.25, 0.75), Pos(0.66, 0.875)),
#         Segment(Pos(0.355, 0.), Pos(0.525, 0.185)),
#         Segment(Pos(0.25, 0.5), Pos(0.75, 0.215)),
#         Segment(Pos(1., 0.25), Pos(0.435, 0.55)),
#     ],
#     "target_pos": (0.15, 0.9),
#     "target_radius": 0.05,
# }

# # Pointmaze maze
# y_bottom = 0.25
# y_top = 0.6
# width = 0.75
# KHERPERAX_MAZES["pointmaze"] = {
#     "segments": [
#         Segment(Pos(0.0, y_bottom), Pos(width, y_bottom)),
#         Segment(Pos(1-width, y_top), Pos(1.0, y_top)),
#     ],
#     "target_pos": (0.9, 0.9),
#     "target_radius": 0.05,
# }


# Curiosity ES mazes
left_wall = 0.2
start_left_max_x = 0.8
start_left_y = [0.8, 0.4]
start_right_min_x = 0.4
start_right_y = [0.6, 0.2]

KHERPERAX_MAZES["snake"] = {
    "segments": [
        Segment(Pos(left_wall, 0.0), Pos(left_wall, 0.8)), # left wall
    ] + [
        Segment(Pos(left_wall, y), Pos(start_left_max_x, y)) for y in start_left_y # walls starting left 
    ] + [
        Segment(Pos(start_right_min_x, y), Pos(1.0, y)) for y in start_right_y # walls starting right 
    ],
    "target_pos": (0.9, 0.1),
    "target_radius": 0.05,
}
