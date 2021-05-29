#! /usr/bin/env python3

import click
import osmnx as osm


@click.command()
@click.option('--center-lat-lng',
              default=(37.7749128, -122.4651554),
              type=tuple)
@click.option('--bbox-size-meters', default=2400, type=float)
@click.option('--output-filename', default='san-francisco.graphml')
@click.option('--include-nodes', is_flag=True)
def main(center_lat_lng, bbox_size_meters, output_filename, include_nodes):
    center = osm.core.bbox_from_point(center_lat_lng,
                                      distance=bbox_size_meters)
    G = osm.core.graph_from_bbox(*center)
    fig, ax = osm.plot_graph(G,
                             show=False,
                             annotate=include_nodes,
                             fig_height=20,
                             fig_width=20)
    plot_filename = output_filename + '.png'
    fig.savefig(plot_filename)
    osm.save_graphml(G, filename=output_filename)


if __name__ == '__main__':
    main()
