from data.loader import FileIO


class MMKEI(object):
    def __init__(self, config):
        self.social_data = []
        self.feature_data = []
        self.config = config
        self.training_data = FileIO.load_data_set(config['training.set'], config['model.type'])
        self.test_data = FileIO.load_data_set(config['test.set'], config['model.type'])

        self.kwargs = {}
        if config.contain('social.data'):
            social_data = FileIO.load_social_data(self.config['social.data'])
            self.kwargs['social.data'] = social_data
        print('Reading data and preprocessing...')

    def execute(self):
        # import the model module
        #import_str = 'from model.'+ self.config['model.type'] +'.' + self.config['model.name'] + ' import ' + self.config['model.name']
        import_str = 'from model.' + self.config['stage'] + '.' + self.config['model.name'] + ' import ' + \
                     self.config['model.name']

        exec(import_str)
        recommender = self.config['model.name'] + '(self.config,self.training_data,self.test_data,**self.kwargs)'
        eval(recommender).execute()
