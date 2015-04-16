'use strict'

###*
 # @ngdoc directive
 # @name myDirectives.directive:myScrollDirective
 # @description
 # # myScrollDirective
###
angular.module('myDirectives', []);

angular.module('myDirectives')
  .directive('myScrollDirective', ->
    restrict: 'A',
    link: (scope, element) ->
  	  # fixes menu after some scrolling
      header = document.getElementById('mainHead')
      sidebar = document.getElementById("sidebar")
      hHeight = header.offsetHeight
      window.onscroll = () ->
      	scrollTop = document.documentElement.scrollTop || document.body.scrollTop;
      	if scrollTop >= hHeight
      	  sidebar.setAttribute('class', 'fixedMenu')
      	else
      	  sidebar.setAttribute('class', '')  
  )
