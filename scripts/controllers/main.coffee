'use strict'

angular.module('angularApp')
	.controller 'MainCtrl', ['$scope', '$rootScope', '$location', ($scope, $rootScope, $location) ->
		$rootScope.profileActive = false
		if $location.path() is "/profile"
			$rootScope.profileActive = true
	]


    
