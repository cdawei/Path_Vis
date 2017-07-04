from django.shortcuts import render, render_to_response
from django.http import HttpResponse, JsonResponse
import sys, cgi, os, json, gzip
import numpy as np
import pickle as pkl

def main(request):
    return render(request, "index.html")

def recommend(request):
    handler = dummyHandler()
    sname = int(request.POST.get("START"))
    length = int(request.POST.get("LENGTH"))
    print("sname: %d, length: %d"%(sname, length))
    data = handler.recommend(sname, length)
    return JsonResponse(data, safe=False)


class dummyHandler():
    def preprocess(self, recommendations):
        # scale scores and convert arrays to lists
        scores_traj = [x['TotalScore'] for x in recommendations]
        score_max = np.max(scores_traj)
        score_min = np.min(scores_traj)
        assert(abs(score_max) > 1e-9)
        assert(abs(score_min) > 1e-9)
        assert(score_max > score_min)

        # linear scaling of trajectory scores
        # a * score_max + b = 100
        # a * score_min + b = 10
        a = np.exp(np.log(90) - np.log(score_max - score_min))
        b = 100 - a * score_max
        #print(a, b)

        # linear scaling of POI feature scores
        # there is a vector of feature scores for each POI in each recommended trajectory
        scores_feature = [x for z in recommendations for y in z['POIPerFeatureScore'] for x in y]  
        score_max_feature = np.max(scores_feature)
        score_min_feature = np.min(scores_feature)
        assert(abs(score_max_feature) > 1e-9)
        #assert(abs(score_min_feature) > 1e-9)
        assert(score_max_feature > score_min_feature)
        # a1 * score_max_feature + b1 = 10
        # a1 * score_min_feature + b1 = 1
        a1 = np.exp(np.log(9) - np.log(score_max_feature - score_min_feature))
        b1 = 10 - a1 * score_max_feature
        #print(a1, b1)
        
        for j in range(len(recommendations)):
            rec = recommendations[j]
            score0 = rec['TotalScore']
            score1 = 0
            if j == 0:
                score1 = 100
            elif j == len(recommendations) - 1:
                score1 = 10
            else:
                score1 = a * rec['TotalScore'] + b

            print('scaled score:', score1)
            assert(score1 > 9)
            assert(score1 < 101)
            recommendations[j]['TotalScore'] = score1

            # Both recommendations[j]['POIFeatureScore'] and recommendations[j]['POIFeatureWeight'] are 24-dimention vectors,
            # and the correspondence between POI features (and feature weights) and elements in these two vectors are:
            # Index    Feature name
            # 0-8      category (POI categories are:
            # [City precincts, Shopping, Entertainment, Public galleries, Institutions, Structures, Sports stadiums, Parks and spaces, Transport])
            # 9-13     neighbourhood
            # 14       popularity
            # 15       nVisit
            # 16       avgDuration
            # 17       trajLen
            # 18       sameCatStart
            # 19       distStart
            # 20       diffPopStart
            # 21       diffNVisitStart
            # 22       diffDurationStart
            # 23       sameNeighbourhoodStart

            # Both recommendations[j]['TransitionFeatureScore'] and recommendations[j]['TransitionFeatureWeight'] are 5-dimentional vectors,
            # and the correspondence between transition features (and feature weights) and elements in these two vectors are:
            # Index    Feature name
            # 0        poiCat        (transition probability according to POI category)
            # 1        popularity    (transition probability according to POI popularity)
            # 2        nVisit        (transition probability according to the number of visit at POI)
            # 3        avgDuration   (transition probability according to the average duration at POI)
            # 4        neighbourhood (transition probability according to the neighbourhood of POI)

            # recommendations[j]['POIPerFeatureScore'][k] is a vector of (feature) scores of the k-th POI in the j-th recommended trajectory

            assert(abs(score0) > 1e-9)
            ratio = np.exp(np.log(score1) - np.log(score0))
            recommendations[j]['POIScore'] = (rec['POIScore'] * ratio).tolist()
            recommendations[j]['TransitionScore'] = (rec['TransitionScore'] * ratio).tolist()
            recommendations[j]['POIFeatureScore'] = (rec['POIFeatureScore'] * ratio).tolist()
            recommendations[j]['TransitionFeatureScore'] = (rec['TransitionFeatureScore'] * ratio).tolist()
            recommendations[j]['Trajectory'] = (rec['Trajectory']).tolist()
            if 'POIFeatureWeight' in rec:
                recommendations[j]['POIFeatureWeight'] = rec['POIFeatureWeight'].tolist()
                recommendations[j]['TransitionFeatureWeight'] = rec['TransitionFeatureWeight'].tolist()
                for k in range(len(rec['Trajectory'])):
                    #recommendations[j]['POIPerFeatureScore'][k] = [x * ratio for x in rec['POIPerFeatureScore'][k]]
                    recommendations[j]['POIPerFeatureScore'][k] = [a1 * x + b1 for x in rec['POIPerFeatureScore'][k]]

        return recommendations


    def recommend(self, start, length):
        #print('in recommend()')
        #startPOI = 9  # the start POI-ID for the desired trajectory, can be any POI-ID in flickr-photo/data/poi-Melb-all.csv
        #length = 8    # the length of desired trajectory: the number of POIs in trajectory (including start POI)
                       # if length > 8, the inference could be slow
        assert(start > 0)
        assert(2 <= length <= 10)

        if not hasattr(self, 'cached_results'):
            data_path = os.path.join(os.path.dirname(__file__), 'data')
            frec = os.path.join(data_path, 'rec-all.gz')
            self.cached_results = pkl.load(gzip.open(frec, 'rb'))
            print('cached results loaded')

        recommendations = self.cached_results[(start, length)]
        for i in range(len(recommendations)):
            print('Top %d recommendation: %s' % (i+1, str(list(recommendations[i]['Trajectory']))))
        for i in range(len(recommendations)):
            print('%s' % recommendations[i]['TotalScore'])

        # return recommended trajectories as well as a number of scores
        return json.dumps(self.preprocess(recommendations), sort_keys=True)
