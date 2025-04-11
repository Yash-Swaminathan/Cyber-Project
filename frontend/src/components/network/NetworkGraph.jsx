import React, { useContext, useEffect, useRef, useState } from 'react';
import { AppContext } from '../../context/AppContext';
import * as d3 from 'd3';

const NetworkGraph = () => {
  const { recentFlows } = useContext(AppContext);
  // Default to empty array if recentFlows is not an array.
  const safeRecentFlows = Array.isArray(recentFlows) ? recentFlows : [];
  
  const svgRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [simulationRunning, setSimulationRunning] = useState(false);

  // Process flow data for visualization.
  const processFlowData = () => {
    const uniqueIps = new Set();
    safeRecentFlows.forEach(flow => {
      uniqueIps.add(flow.src_ip);
      uniqueIps.add(flow.dst_ip);
    });
    
    const nodes = Array.from(uniqueIps).map(ip => ({
      id: ip,
      group: ip.startsWith('192.168') ? 1 : ip.startsWith('10.') ? 2 : 3
    }));
    
    const links = safeRecentFlows.map(flow => ({
      source: flow.src_ip,
      target: flow.dst_ip,
      value: (flow.bytes_sent || 0) + (flow.bytes_received || 0),
      protocol: flow.protocol
    }));
    
    return { nodes, links };
  };

  // Initialize and update the graph visualization.
  useEffect(() => {
    if (safeRecentFlows.length === 0 || !svgRef.current) return;

    const graphData = processFlowData();
    d3.select(svgRef.current).selectAll("*").remove();
    const svg = d3.select(svgRef.current)
      .attr("width", dimensions.width)
      .attr("height", dimensions.height)
      .attr("viewBox", [0, 0, dimensions.width, dimensions.height])
      .attr("style", "max-width: 100%; height: auto;");

    const protocolColor = (protocol) => {
      switch (protocol) {
        case 'TCP': return "#4299e1";
        case 'UDP': return "#48bb78";
        case 'ICMP': return "#ed8936";
        default: return "#a0aec0";
      }
    };

    const simulation = d3.forceSimulation(graphData.nodes)
      .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(100))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(dimensions.width / 2, dimensions.height / 2))
      .force("collision", d3.forceCollide().radius(30));

    const link = svg.append("g")
      .selectAll("line")
      .data(graphData.links)
      .join("line")
      .attr("stroke", d => protocolColor(d.protocol))
      .attr("stroke-opacity", 0.6)
      .attr("stroke-width", d => Math.sqrt(d.value) / 500 + 1);

    const node = svg.append("g")
      .selectAll("circle")
      .data(graphData.nodes)
      .join("circle")
      .attr("r", 10)
      .attr("fill", d => d.group === 1 ? "#7f9cf5" : d.group === 2 ? "#f687b3" : "#e53e3e")
      .call(drag(simulation));

    node.append("title").text(d => d.id);

    const label = svg.append("g")
      .selectAll("text")
      .data(graphData.nodes)
      .join("text")
      .attr("dx", 12)
      .attr("dy", ".35em")
      .text(d => d.id)
      .style("font-size", "10px");

    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);
      
      node
        .attr("cx", d => d.x = Math.max(10, Math.min(dimensions.width - 10, d.x)))
        .attr("cy", d => d.y = Math.max(10, Math.min(dimensions.height - 10, d.y)));
      
      label
        .attr("x", d => d.x)
        .attr("y", d => d.y);
    });
    
    setSimulationRunning(true);
    return () => {
      simulation.stop();
      setSimulationRunning(false);
    };
  }, [safeRecentFlows, dimensions]);

  // Drag behavior for nodes.
  const drag = (simulation) => {
    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }
    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }
    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
    return d3.drag()
      .on("start", dragstarted)
      .on("drag", dragged)
      .on("end", dragended);
  };

  // Handle resizing
  useEffect(() => {
    const updateDimensions = () => {
      const container = svgRef.current?.parentElement;
      if (container) {
        setDimensions({
          width: container.clientWidth,
          height: Math.max(500, window.innerHeight * 0.7)
        });
      }
    };
    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  return (
    <div className="p-6">
      <h2 className="text-2xl font-semibold mb-4">Network Traffic Graph</h2>
      <div className="bg-white p-4 rounded-lg shadow-md mb-4">
        <div className="flex justify-between items-center mb-4">
          <div>
            {(() => {
              const { nodes, links } = processFlowData();
              return (
                <>
                  <span className="font-medium">Total Nodes:</span> {nodes.length}
                  <span className="ml-4 font-medium">Total Connections:</span> {links.length}
                </>
              );
            })()}
          </div>
          <div>
            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
              simulationRunning ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
            }`}>
              {simulationRunning ? 'Simulation Active' : 'Simulation Paused'}
            </span>
          </div>
        </div>
        <div className="legend flex gap-4 text-sm mb-4">
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-blue-500 mr-1"></div>
            <span>TCP</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-green-500 mr-1"></div>
            <span>UDP</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-orange-500 mr-1"></div>
            <span>ICMP</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-indigo-400 mr-1"></div>
            <span>Internal IP</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-pink-400 mr-1"></div>
            <span>Private IP</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-red-600 mr-1"></div>
            <span>External IP</span>
          </div>
        </div>
        <div className="border rounded-lg overflow-hidden">
          <svg ref={svgRef} className="w-full"></svg>
        </div>
        <div className="text-xs text-gray-500 mt-2">
          Tip: Drag nodes to rearrange the network graph
        </div>
      </div>
      {safeRecentFlows.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          No network flow data available. Wait for data to arrive or check your connection.
        </div>
      )}
    </div>
  );
};

export default NetworkGraph;
