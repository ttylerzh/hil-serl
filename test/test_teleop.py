from ur7e_env.spacemouse.teleop import parse_args


def test_teleop_task_argument():
    assert parse_args([])[1].task == "plug_insertion"
    assert parse_args(["pick"])[1].task == "pick"


if __name__ == "__main__":
    test_teleop_task_argument()
