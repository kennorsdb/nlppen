import datetime

class ProcessException(Exception):
    def __init__(self, proceso, error):
        super(ProcessException, self).__init__(proceso, error )
        self.error = {"timestamp": datetime.datetime.utcnow(), "proceso": proceso, "error": error}

    def get_error(self):
        return self.error

    