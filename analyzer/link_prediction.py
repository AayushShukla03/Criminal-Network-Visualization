import sys
import os
import networkx.algorithms.link_prediction as methods
import networkx.algorithms.community as community_methods
from operator import itemgetter
import networkx as nx


# find path to root directory of the project so as to import from other packages
tokens = os.path.abspath(__file__).split('/')
# print('tokens = ', tokens)
path2root = '/'.join(tokens[:-2])
# print('path2root = ', path2root)
if path2root not in sys.path:
    sys.path.append(path2root)

import analyzer.common.helpers as helpers


def _get_sources(nx_graph, params, node_index):
    if 'sources' in params:
        sources = params['sources']
        sources = [node_index[u] for u in sources]
    else:
        sources = nx_graph.nodes

    return sources


def _get_candidates(nx_graph, sources):
    """
    find candidate (new) links for a list of nodes
    :param network: networkx network
    :param sources: list of node id
    :return:
    """
    candidates = []

    if sources is None:
        sources = nx_graph.nodes
    # TODO: to add more selection for identifying the candidates
    for u in sources:
        neighbors = nx_graph.neighbors(u)
        second_hop_neighbors = set()
        for v in neighbors:
            second_hop_neighbors = second_hop_neighbors.union(set(nx_graph.neighbors(v)))
        second_hop_neighbors = second_hop_neighbors.difference(neighbors)
        if u in second_hop_neighbors:
            second_hop_neighbors.remove(u)
        candidates.extend([(u, v) for v in second_hop_neighbors])

    return candidates


def _select_top_k(candidates, k=3):
    candidates.sort(key=itemgetter(1), reverse=True)
    return [u[0] for u in candidates[:k]]


def _generate_link_predictions(scores, params, sources, node_ids):
    preds = dict([(node_ids[u], []) for u in sources])

    for u, v, p in scores:
        # print('(%d, %d) -> %.8f' % (u, v, p))
        preds[node_ids[u]].append((node_ids[v], p))

    for u in preds:
        if 'top_k' in params:
            preds[u] = _select_top_k(preds[u], params['top_k'])
        else:
            preds[u] = _select_top_k(preds[u])

    return preds


def _call_nx_community_detection_method(method_name, graph):
    """
    call networkx' community detection methods. 
    supported methods including 'modularity', 'asyn_lpa', 'label_propagation'
    :param method_name: the name of networkx' community detection algorithm
    :param graph: networkx graph
    :return:
    """
    supported_community_methods = ('modularity', 'asyn_lpa', 'label_propagation')

    if method_name not in supported_community_methods:
        return None

    if method_name is supported_community_methods[0]:
        return community_methods.greedy_modularity_communities(graph)
    elif method_name is supported_community_methods[1]:
        return community_methods.asyn_lpa_communities(graph)
    elif method_name is supported_community_methods[2]:
        return community_methods.label_propagation_communities(graph)
    else:
        return None


def resource_allocation_index(network, params):
    """
    predict links for a set of nodes using networkx' resource_allocation_index function
    :param network: networkx network
    :param params:
    :return: dictionary, in the form
        {
            'success': 1 if success, 0 otherwise
            'message': a string
            'predictions': predictions
        }
    """
    try:
        graph, node_ids = helpers.convert_to_nx_undirected_graph(network)
        node_index = [(node_ids[i], i) for i in range(len(node_ids))]
        node_index = dict(node_index)
        if params is None:
            params = {}

        sources = _get_sources(graph, params, node_index)
        candidates = _get_candidates(graph, sources)
        scores = methods.resource_allocation_index(graph, candidates)
        predictions = _generate_link_predictions(scores, params, sources, node_ids)
        result = {'success': 1, 'message': 'the task is performed successfully', 'predictions': predictions}
        return result
    except Exception as e:
        print(e)
        result = {'success': 0, 'message': 'this algorithm is not suitable for the input network', 'predictions': None}
        return result


def jaccard_coefficient(network, params=None):
    """
    predict links for a set of nodes using networkx' jaccard_coefficient function
    :param network: networkx network
    :param params:
    :return: dictionary, in the form
        {
            'success': 1 if success, 0 otherwise
            'message': a string
            'predictions': predictions
        }
    """
    try:
        graph, node_ids = helpers.convert_to_nx_undirected_graph(network)
        node_index = [(node_ids[i], i) for i in range(len(node_ids))]
        node_index = dict(node_index)
        if params is None:
            params = {}

        sources = _get_sources(graph, params, node_index)
        candidates = _get_candidates(graph, sources)
        scores = methods.jaccard_coefficient(graph, candidates)
        predictions = _generate_link_predictions(scores, params, sources, node_ids)

        result = {'success': 1, 'message': 'the task is performed successfully', 'predictions': predictions}
        return result
    except Exception as e:
        print(e)
        result = {'success': 0, 'message': 'this algorithm is not suitable for the input network', 'predictions': None}
        return result


def adamic_adar_index(network, params):
    """
    predict links for a set of nodes using networkx' adamic_adar_index function
    :param network: networkx network
    :param params:
    :return: dictionary, in the form
        {
            'success': 1 if success, 0 otherwise
            'message': a string
            'predictions': predictions
        }
    """
    try:
        graph, node_ids = helpers.convert_to_nx_undirected_graph(network)
        node_index = [(node_ids[i], i) for i in range(len(node_ids))]
        node_index = dict(node_index)
        if params is None:
            params = {}

        sources = _get_sources(graph, params, node_index)
        candidates = _get_candidates(graph, sources)
        scores = methods.adamic_adar_index(graph, candidates)
        predictions = _generate_link_predictions(scores, params, sources, node_ids)

        result = {'success': 1, 'message': 'the task is performed successfully', 'predictions': predictions}
        return result
    except Exception as e:
        print(e)
        result = {'success': 0, 'message': 'this algorithm is not suitable for the input network', 'predictions': None}
        return result


def preferential_attachment(network, params):
    """
    predict links for a set of nodes using networkx' preferential_attachment function to
    compute the preferential attachment score of all node pairs in network.
    :param network: networkx network
    :param params:
    :return: dictionary, in the form
        {
            'success': 1 if success, 0 otherwise
            'message': a string
            'predictions': predictions
        }
    """
    try:
        graph, node_ids = helpers.convert_to_nx_undirected_graph(network)
        node_index = [(node_ids[i], i) for i in range(len(node_ids))]
        node_index = dict(node_index)
        if params is None:
            params = {}

        sources = _get_sources(graph, params, node_index)
        candidates = _get_candidates(graph, sources)
        scores = methods.preferential_attachment(graph, candidates)
        predictions = _generate_link_predictions(scores, params, sources, node_ids)

        result = {'success': 1, 'message': 'the task is performed successfully', 'predictions': predictions}
        return result
    except Exception as e:
        print(e)
        result = {'success': 0, 'message': 'this algorithm is not suitable for the input network', 'predictions': None}
        return result


def count_number_soundarajan_hopcroft(network, params):
    """
    predict links for a set of nodes using networkx' cn_soundarajan_hopcroft function to 
    count the number of common neighbors
    :param network: networkx network
    :param params:
    :return: dictionary, in the form
        {
            'success': 1 if success, 0 otherwise
            'message': a string
            'predictions': predictions
        }
    """
    try:
        graph, node_ids = helpers.convert_to_nx_undirected_graph(network)
        node_index = [(node_ids[i], i) for i in range(len(node_ids))]
        node_index = dict(node_index)
        if params is None:
            params = {}

        try:
            community_detection_method = params['community_detection_method']
        except Exception as e:
            print('Community detection method is not defined.', e)
            return None

        nx_comms = _call_nx_community_detection_method(community_detection_method, graph)
        if nx_comms is None:
            print("Community detection method is not supported.")
            return None
        else:
            nx_comms = list(nx_comms)

        # initalize community information
        for node in graph.nodes():
            graph.nodes[node]['community'] = None

        # add community information
        for i in range(len(nx_comms)):
            for node in nx_comms[i]:
                if graph.nodes[node]['community'] is None:
                    graph.nodes[node]['community'] = i

        sources = _get_sources(graph, params, node_index)
        candidates = _get_candidates(graph, sources)
        scores = methods.cn_soundarajan_hopcroft(graph, candidates)
        predictions = _generate_link_predictions(scores, params, sources, node_ids)

        result = {'success': 1, 'message': 'the task is performed successfully', 'predictions': predictions}
        return result
    except Exception as e:
        print(e)
        result = {'success': 0, 'message': 'this algorithm is not suitable for the input network', 'predictions': None}
        return result


def resource_allocation_index_soundarajan_hopcroft(network, params):
    """
    predict links for a set of nodes using networkx' ra_index_soundarajan_hopcroft function to 
    compute the resource allocation index of all node pairs in network using community information.
    :param network: networkx network
    :param params:
    :return: dictionary, in the form
        {
            'success': 1 if success, 0 otherwise
            'message': a string
            'predictions': predictions
        }
    """
    try:
        graph, node_ids = helpers.convert_to_nx_undirected_graph(network)
        node_index = [(node_ids[i], i) for i in range(len(node_ids))]
        node_index = dict(node_index)
        if params is None:
            params = {}

        try:
            community_detection_method = params['community_detection_method']
        except Exception as e:
            print('Community detection method is not defined.', e)
            return None

        nx_comms = _call_nx_community_detection_method(community_detection_method, graph)
        if nx_comms is None:
            print("Community detection method is not supported.")
            return None
        else:
            nx_comms = list(nx_comms)

        # initalize community information
        for node in graph.nodes():
            graph.nodes[node]['community'] = None

        # add community information
        for i in range(len(nx_comms)):
            for node in nx_comms[i]:
                if graph.nodes[node]['community'] is None:
                    graph.nodes[node]['community'] = i

        sources = _get_sources(graph, params, node_index)
        candidates = _get_candidates(graph, sources)
        scores = methods.ra_index_soundarajan_hopcroft(graph, candidates)
        predictions = _generate_link_predictions(scores, params, sources, node_ids)

        result = {'success': 1, 'message': 'the task is performed successfully', 'predictions': predictions}
        return result
    except Exception as e:
        print(e)
        result = {'success': 0, 'message': 'this algorithm is not suitable for the input network', 'predictions': None}
        return result


def within_inter_cluster(network, params):
    """
    predict links for a set of nodes using networkx' within_inter_cluster to 
    compute the ratio of within- and inter-cluster common neighbors of all node pairs in network.
    :param network: networkx network
    :param params:
    :return: dictionary, in the form
        {
            'success': 1 if success, 0 otherwise
            'message': a string
            'predictions': predictions
        }
    """
    try:
        graph, node_ids = helpers.convert_to_nx_undirected_graph(network)
        node_index = [(node_ids[i], i) for i in range(len(node_ids))]
        node_index = dict(node_index)
        if params is None:
            params = {}

        try:
            community_detection_method = params['community_detection_method']
        except Exception as e:
            print('Community detection method is not defined.', e)
            return None

        nx_comms = _call_nx_community_detection_method(community_detection_method, graph)
        if nx_comms is None:
            print("Community detection method is not supported.")
            return None
        else:
            nx_comms = list(nx_comms)

        # initalize community information
        for node in graph.nodes():
            graph.nodes[node]['community'] = None

        # add community information
        for i in range(len(nx_comms)):
            for node in nx_comms[i]:
                if graph.nodes[node]['community'] is None:
                    graph.nodes[node]['community'] = i

        sources = _get_sources(graph, params, node_index)
        candidates = _get_candidates(graph, sources)
        scores = methods.within_inter_cluster(graph, candidates)
        predictions = _generate_link_predictions(scores, params, sources, node_ids)

        result = {'success': 1, 'message': 'the task is performed successfully', 'predictions': predictions}
        return result
    except Exception as e:
        print(e)
        result = {'success': 0, 'message': 'this algorithm is not suitable for the input network', 'predictions': None}
        return result


def get_info():
    """
    get information about methods provided in this class
    :return: dictionary: Provides the name of the analysis task, available methods and information
                         about an methods parameter. Also provides full names of tasks, methods and parameter.
                         Information is provided in the following format:

                        {
                            'name': Full analysis task name as string
                            'methods': {
                                key: Internal method name (eg. 'asyn_lpa')
                                value: {
                                    'name': Full method name as string
                                    'parameter': {
                                        key: Parameter name
                                        value: {
                                            'description': Description of the parameter
                                            'options': {
                                                key: Accepted parameter value
                                                value: Full parameter value name as string
                                                !! If accepted values are integers key and value is 'Integer'. !!
                                            }
                                        }
                                    }
                                }
                            }
                        }

    """
    info = {'name': 'Link Prediction',
            'methods': {
                'resource_allocation_index': {
                    'name': 'Resource Allocation Index',
                    'parameter': {}
                },
                'jaccard_coefficient': {
                    'name': 'Jaccard Coefficient',
                    'parameter': {}
                },
                'adamic_adar_index': {
                    'name': 'Adamic Adar Index',
                    'parameter': {}
                },
                'count_number_soundarajan_hopcroft': {
                    'name': 'Soundarajan Hopcroft (Count Numbers)',
                    'parameter': {
                        'community_detection_method': {
                            'description': 'Community detection method',
                            'options': {'modularity': 'Modularity',
                                        'asyn_lpa': 'Asynchronous Label Propagation',
                                        'label_propagation': 'Label Propagation'}
                        }
                    }
                },
                'resource_allocation_index_soundarajan_hopcroft': {
                    'name': 'Resource Alocation Index (Soundarajan Hopcroft)',
                    'parameter': {
                        'community_detection_method': {
                            'description': 'Community detection method',
                            'options': {'modularity': 'Modularity',
                                        'asyn_lpa': 'Asynchronous Label Propagation',
                                        'label_propagation': 'Label Propagation'}
                        }
                    }
                },
                'within_inter_cluster': {
                    'name': 'Within- and Interclustering',
                    'parameter': {
                        'community_detection_method': {
                            'description': 'Community detection method',
                            'options': {'modularity': 'Modularity',
                                        'asyn_lpa': 'Asynchronous Label Propagation',
                                        'label_propagation': 'Label Propagation'}
                        }
                    }
                }
            }
            }
    return info


class LinkPredictor:
    """
    class for performing link prediction
    """

    def __init__(self, algorithm):
        """
        init a community detector using the given `algorithm`
        :param algorithm:
        """
        self.algorithm = algorithm
        self.methods = {
            'resource_allocation_index': resource_allocation_index,
            'jaccard_coefficient': jaccard_coefficient,
            'adamic_adar_index': adamic_adar_index,
            'preferential_attachment': preferential_attachment,
            'count_number_soundarajan_hopcroft': count_number_soundarajan_hopcroft,
            'resource_allocation_index_soundarajan_hopcroft': resource_allocation_index_soundarajan_hopcroft,
            'within_inter_cluster': within_inter_cluster
            # TODO: to add more methods from networkx, snap, and sklearn
        }

    def perform(self, network, params):
        """
        performing
        :param network:
        :param params:
        :return:
        """
        
        result = self.methods[self.algorithm](network, params)

        # Metric 1: Count the number of crimes
        predicted_crime_count = len([key for key in result['predictions'].keys() if key.startswith('crime')])
        print("1. Predicted number of crimes:", predicted_crime_count)

        # Metric 2: Count the number of predicted links in total
        predicted_link_count = sum(len(links) for links in result['predictions'].values())
        print("2. Total predicted links:", predicted_link_count)

        # Metric 3: Count the number of unique nodes involved in predictions
        predicted_nodes = set(node for links in result['predictions'].values() for node in links)
        predicted_node_count = len(predicted_nodes)
        print("3. Number of unique nodes involved in predictions:", predicted_node_count)

        # Metric 4: Calculate the average number of predictions per node
        average_predictions_per_node = predicted_link_count / predicted_node_count if predicted_node_count != 0 else 0
        print("4. Average predictions per node:", average_predictions_per_node)

        # Metric 5: Count the number of nodes involved in crimes
        predicted_crime_nodes = set(key.split('_')[1] for key in result['predictions'].keys() if key.startswith('crime'))
        predicted_crime_node_count = len(predicted_crime_nodes)
        print("5. Number of nodes involved in predicted crimes:", predicted_crime_node_count)

        # Metric 6: Calculate the average number of predicted crimes per crime node
        average_predicted_crimes_per_node = predicted_crime_count / predicted_crime_node_count if predicted_crime_node_count != 0 else 0
        print("6. Average predicted crimes per crime node:", average_predicted_crimes_per_node)

        # Metric 7: Calculate the average number of predictions per crime
        average_predictions_per_crime = predicted_link_count / predicted_crime_count if predicted_crime_count != 0 else 0
        print("7. Average predictions per crime:", average_predictions_per_crime)

        # Metric 8: Count the number of crimes involved in predictions
        predicted_crime_keys = [key for key in result['predictions'].keys() if key.startswith('crime')]
        print("8. Number of crimes involved in predictions:", len(predicted_crime_keys))

        # Metric 9: Calculate the average number of predicted links per crime
        average_links_per_crime = predicted_link_count / len(predicted_crime_keys) if len(predicted_crime_keys) != 0 else 0
        print("9. Average predicted links per crime:", average_links_per_crime)


        if result['success'] == 1:
            # Extract predicted graph from result
            predicted_graph = nx.Graph(result['predictions'])

            # Calculate betweenness centrality values
            betweenness_centrality_values = nx.betweenness_centrality(predicted_graph)

            # Display betweenness centrality values
            print("Betweenness centrality values:")
            for node_id, centrality in betweenness_centrality_values.items():
                print(f"{node_id}: {centrality}")

            # Metric 10: Calculate the average betweenness centrality
            average_betweenness_centrality = sum(betweenness_centrality_values.values()) / len(betweenness_centrality_values)
            print("10. Average betweenness centrality:", average_betweenness_centrality)
    
            # Calculate closeness centrality values
            closeness_centrality_values = nx.closeness_centrality(predicted_graph)

            # Display closeness centrality values
            print("Closeness centrality values:")
            for node_id, centrality in closeness_centrality_values.items():
                print(f"{node_id}: {centrality}")

            # Metric 11: Calculate the average closeness centrality
            average_closeness_centrality = sum(closeness_centrality_values.values()) / len(closeness_centrality_values)
            print("11. Average closeness centrality:", average_closeness_centrality)

            # Calculate eigenvector centrality values
            eigenvector_centrality_values = nx.eigenvector_centrality(predicted_graph)

            # Display eigenvector centrality values
            print("Eigenvector centrality values:")
            for node_id, centrality in eigenvector_centrality_values.items():
                print(f"{node_id}: {centrality}")

            # Metric 12: Calculate the average eigenvector centrality
            average_eigenvector_centrality = sum(eigenvector_centrality_values.values()) / len(eigenvector_centrality_values)
            print("12. Average eigenvector centrality:", average_eigenvector_centrality)

            # Metric 13: Count the number of unique nodes involved in predictions
            predicted_node_count = len(predicted_graph.nodes)
            print("13. Number of unique nodes involved in predictions:", predicted_node_count)

            # Metric 14: Calculate the average number of predictions per node
            predicted_link_count = len(predicted_graph.edges)
            average_predictions_per_node = predicted_link_count / predicted_node_count if predicted_node_count != 0 else 0
            print("14. Average predictions per node:", average_predictions_per_node)

            # Metric 15: Count the total number of predicted links
            print("15. Total predicted links:", predicted_link_count)

            # Metric 16: Count the number of crimes involved in predictions
            predicted_crime_count = sum(1 for node_id in predicted_graph.nodes if node_id.startswith('crime'))
            print("16. Predicted number of crimes:", predicted_crime_count)

            # Metric 17: Calculate the average number of predicted links per crime
            predicted_crime_nodes = [node_id for node_id in predicted_graph.nodes if node_id.startswith('crime')]
            average_links_per_crime = predicted_link_count / len(predicted_crime_nodes) if len(predicted_crime_nodes) != 0 else 0
            print("17. Average predicted links per crime:", average_links_per_crime)
            

        return self.methods[self.algorithm](network, params)

