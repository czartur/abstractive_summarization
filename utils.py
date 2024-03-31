import argparse 
from dataclasses import fields
from typing import get_origin, Union 

def dataclass_to_argparse(dc):
    parser = argparse.ArgumentParser()
    for dc_field in fields(dc):
        field_type = dc_field.type
        field_name = dc_field.name.replace('_', '-')
        field_default = dc_field.default
        if field_type is bool:
            parser.add_argument(
                f'--{field_name}',
                action='store_true',
                help=f'{field_name} (default: {field_default})'
            )
            parser.add_argument(
                f'--no-{field_name}',
                dest=field_name,
                action='store_false'
            )
            parser.set_defaults(**{field_name: field_default})
        elif get_origin(field_type) == Union:
            field_types = field_type.__args__
            type_lambda = lambda x: next((t(x) for t in field_types if isinstance(x, t)), None)
            parser.add_argument(
                f'--{field_name}',
                type=type_lambda,
                default=field_default,
                help=f'{field_name} (default: {field_default})'
            )
        else:
            parser.add_argument(
                f'--{field_name}',
                type=field_type,
                default=field_default,
                help=f'{field_name} (default: {field_default})'
            )
    return parser

def parse_args_to_dataclass(dc_cls):
    parser = dataclass_to_argparse(dc_cls)
    args = parser.parse_args()
    return dc_cls(**vars(args))