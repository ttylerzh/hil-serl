from experiments.ram_insertion.config import TrainConfig as RAMInsertionTrainConfig
from experiments.usb_pickup_insertion.config import TrainConfig as USBPickupInsertionTrainConfig
from experiments.object_handover.config import TrainConfig as ObjectHandoverTrainConfig
from experiments.egg_flip.config import TrainConfig as EggFlipTrainConfig
from experiments.pick.config import TrainConfig as PickTrainConfig
from experiments.plug_insertion.config import TrainConfig as PlugInsertionTrainConfig
from experiments.remove_sfp.config import TrainConfig as RemoveSFPTrainConfig

CONFIG_MAPPING = {
                "ram_insertion": RAMInsertionTrainConfig,
                "usb_pickup_insertion": USBPickupInsertionTrainConfig,
                "object_handover": ObjectHandoverTrainConfig,
                "egg_flip": EggFlipTrainConfig,
                "pick": PickTrainConfig,
                "plug_insertion": PlugInsertionTrainConfig,
                "remove_sfp": RemoveSFPTrainConfig,
               }
