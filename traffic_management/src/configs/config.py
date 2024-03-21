import warnings


class Config:

    @classmethod
    def get_globals(cls):
        static_var_vals = {}
        for attr_name in dir(cls):
            if not attr_name.startswith('_'):
                attr = getattr(cls, attr_name)

                if not callable(attr):
                    static_var_vals[attr_name] = attr

        return static_var_vals

    @classmethod
    def update_globals(cls, modifications):
        for attr_name, attr in modifications.items():
            setattr(cls, attr_name, attr)

    @classmethod
    def warn_unknown_configuration(cls, k, v):
        warnings.warn(f"Unknown configuration ({cls.__name__})-> \n\tK:{k}\n\tV:{v}")

    @classmethod
    def warn_missing_configuration(cls, k, v):
        warnings.warn(f"Missing configuration. Using Default. ({cls.__name__})-> \n\tK:{k}\n\tV:{v}")
