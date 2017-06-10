var selectionRect = {
	parentElement 	: null,
	element			: null,
	currentY		: 0,
	currentX		: 0,
	originX			: 0,
	originY			: 0,

	    var imageset_width = 960 - margin.left - margin.right;
	    var imageset_height = 500 - (margin.top + height) - margin.bottom;

	    var imageset = d3.select("#imageset").append("svg")
	        .attr("width", imageset_width + margin.left + margin.right)
	        .attr("height", imageset_height + margin.top + margin.bottom)
	        .attr("align", "center")
	      .append("g")
	        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

